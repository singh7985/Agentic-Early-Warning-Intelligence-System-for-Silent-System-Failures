"""
Log Parsing & Normalization Module

Parses unstructured logs (HDFS, BGL, etc.) and converts them into
structured incident narratives for RAG knowledge base.

Supports:
- Template-based log parsing
- Message normalization
- Incident grouping
- Synthetic maintenance report generation
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json

import pandas as pd

logger = logging.getLogger(__name__)


class LogParser:
    """
    Parses and normalizes system logs.
    """
    
    def __init__(self):
        """Initialize log parser with common patterns."""
        # Common patterns for log parsing
        self.timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
            r'\d{2}/\w{3}/\d{4} \d{2}:\d{2}:\d{2}',  # DD/MMM/YYYY HH:MM:SS
            r'\w{3} +\d{1,2} \d{2}:\d{2}:\d{2}',     # MMM DD HH:MM:SS
        ]
        
        # Error/exception patterns
        self.error_keywords = [
            'error', 'exception', 'failed', 'failure', 'critical',
            'fatal', 'panic', 'warn', 'warning', 'fault'
        ]
    
    def extract_log_fields(self, log_line: str) -> Dict[str, str]:
        """
        Extract structured fields from a log line.
        
        Args:
            log_line: Raw log line
            
        Returns:
            Dictionary with extracted fields (timestamp, level, message, etc.)
        """
        fields = {
            'timestamp': None,
            'level': None,
            'source': None,
            'message': None,
            'raw': log_line
        }
        
        # Extract timestamp
        for pattern in self.timestamp_patterns:
            match = re.search(pattern, log_line)
            if match:
                fields['timestamp'] = match.group(0)
                break
        
        # Extract log level
        level_match = re.search(r'\b(DEBUG|INFO|WARN|ERROR|FATAL|CRITICAL)\b', log_line, re.IGNORECASE)
        if level_match:
            fields['level'] = level_match.group(1).upper()
        
        # Extract source (host, component, etc.)
        source_match = re.search(r'(\w+)[:\[\s]', log_line)
        if source_match:
            fields['source'] = source_match.group(1)
        
        # Extract message (everything after level/source)
        message_start = max(
            log_line.find(fields['level']) if fields['level'] else 0,
            log_line.find(fields['source']) if fields['source'] else 0
        )
        if message_start > 0:
            fields['message'] = log_line[message_start:].strip()
        else:
            fields['message'] = log_line
        
        return fields
    
    def normalize_message(self, message: str) -> str:
        """
        Normalize log message (remove IDs, IP addresses, etc.).
        
        Args:
            message: Raw log message
            
        Returns:
            Normalized message
        """
        # Remove IP addresses
        message = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', message)
        
        # Remove numeric IDs (but keep some context)
        message = re.sub(r'\b(?:pid|process_id|task_id)[\s=:]*\d+', '<PID>', message)
        
        # Remove UUIDs
        message = re.sub(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '<UUID>',
            message,
            flags=re.IGNORECASE
        )
        
        # Remove file paths (keep just filename)
        message = re.sub(r'/[\w/.-]+/(\w+\.log)', r'\1', message)
        
        # Remove long hex strings
        message = re.sub(r'\b[0-9a-f]{32,}\b', '<HASH>', message, flags=re.IGNORECASE)
        
        # Lowercase
        message = message.lower()
        
        return message.strip()
    
    def parse_log_file(self, file_path: str) -> pd.DataFrame:
        """
        Parse a log file into a structured dataframe.
        
        Args:
            file_path: Path to log file
            
        Returns:
            Dataframe with parsed logs
        """
        logs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.strip():
                        parsed = self.extract_log_fields(line)
                        parsed['normalized_message'] = self.normalize_message(
                            parsed['message'] or ''
                        )
                        logs.append(parsed)
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return pd.DataFrame()
        
        df = pd.DataFrame(logs)
        logger.info(f"Parsed {len(df)} log entries from {file_path}")
        
        return df


class IncidentGrouper:
    """
    Groups related logs into incidents/anomalies.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize grouper.
        
        Args:
            window_size: Number of consecutive error logs to group together
        """
        self.window_size = window_size
    
    def group_by_error_bursts(self, df: pd.DataFrame) -> List[Dict]:
        """
        Group logs into error bursts (consecutive errors).
        
        Args:
            df: Parsed logs dataframe
            
        Returns:
            List of incident dictionaries
        """
        incidents = []
        
        # Filter for errors
        error_keywords = ['error', 'exception', 'failed', 'failure', 'fatal']
        error_df = df[
            df['message'].str.lower().str.contains(
                '|'.join(error_keywords),
                na=False
            )
        ].reset_index(drop=True)
        
        if len(error_df) == 0:
            return incidents
        
        # Group consecutive errors
        error_indices = error_df.index.tolist()
        current_burst = [error_indices[0]]
        
        for i in range(1, len(error_indices)):
            # If within window, add to burst
            if error_indices[i] - error_indices[i-1] <= self.window_size:
                current_burst.append(error_indices[i])
            else:
                # Save current burst
                if current_burst:
                    incidents.append({
                        'incident_id': len(incidents),
                        'log_indices': current_burst,
                        'start_idx': current_burst[0],
                        'end_idx': current_burst[-1],
                        'num_errors': len(current_burst),
                        'messages': error_df.loc[current_burst, 'normalized_message'].tolist()
                    })
                current_burst = [error_indices[i]]
        
        # Save last burst
        if current_burst:
            incidents.append({
                'incident_id': len(incidents),
                'log_indices': current_burst,
                'start_idx': current_burst[0],
                'end_idx': current_burst[-1],
                'num_errors': len(current_burst),
                'messages': error_df.loc[current_burst, 'normalized_message'].tolist()
            })
        
        logger.info(f"Identified {len(incidents)} incidents from {len(error_df)} error logs")
        
        return incidents
    
    @staticmethod
    def create_incident_narrative(incident: Dict) -> str:
        """
        Create a human-readable narrative from an incident.
        
        Args:
            incident: Incident dictionary
            
        Returns:
            Narrative string
        """
        narrative = f"""
Incident #{incident['incident_id']}
Duration: Logs {incident['start_idx']} to {incident['end_idx']}
Error Count: {incident['num_errors']}

Error Messages:
"""
        for i, msg in enumerate(incident['messages'][:5], 1):
            narrative += f"\n{i}. {msg}"
        
        if len(incident['messages']) > 5:
            narrative += f"\n... and {len(incident['messages']) - 5} more errors"
        
        return narrative


class SyntheticReportGenerator:
    """
    Generates synthetic maintenance reports from incidents.
    """
    
    TEMPLATES = {
        'connection_failure': (
            "Connection failure detected on {component}. "
            "Root cause: Network timeout or service unavailable. "
            "Recommended action: Check network connectivity and service status."
        ),
        'memory_error': (
            "Memory allocation failure on {component}. "
            "Out of memory condition detected. "
            "Recommended action: Increase memory allocation or optimize memory usage."
        ),
        'disk_error': (
            "Disk I/O error on {component}. "
            "File system may be degraded. "
            "Recommended action: Check disk health and file system integrity."
        ),
        'timeout': (
            "Operation timeout on {component}. "
            "Process took longer than expected. "
            "Recommended action: Increase timeout threshold or optimize performance."
        ),
        'generic': (
            "System anomaly detected on {component}. "
            "Multiple error logs indicate potential degradation. "
            "Recommended action: Escalate to operations team for investigation."
        )
    }
    
    @staticmethod
    def generate_report(incident: Dict, logs_df: pd.DataFrame) -> str:
        """
        Generate a synthetic maintenance report.
        
        Args:
            incident: Incident dictionary
            logs_df: Full logs dataframe
            
        Returns:
            Report text
        """
        # Determine report type from messages
        messages = ' '.join(incident['messages']).lower()
        
        report_type = 'generic'
        if 'connection' in messages or 'timeout' in messages or 'network' in messages:
            report_type = 'connection_failure'
        elif 'memory' in messages or 'oom' in messages:
            report_type = 'memory_error'
        elif 'disk' in messages or 'io error' in messages:
            report_type = 'disk_error'
        elif 'timeout' in messages:
            report_type = 'timeout'
        
        # Extract component (host/source)
        component = logs_df.iloc[incident['log_indices'][0]]['source'] or 'system'
        
        # Generate report
        template = SyntheticReportGenerator.TEMPLATES.get(report_type, SyntheticReportGenerator.TEMPLATES['generic'])
        report = template.format(component=component)
        
        # Add incident context
        report += f"\n\nIncident Details:\n"
        report += f"- Affected Component: {component}\n"
        report += f"- Error Count: {incident['num_errors']}\n"
        report += f"- Incident Type: {report_type}\n"
        
        return report


def load_and_parse_logs(
    log_file_path: str,
    group_incidents: bool = True
) -> Tuple[pd.DataFrame, Optional[List[Dict]]]:
    """
    Convenience function to load and parse logs.
    
    Args:
        log_file_path: Path to log file
        group_incidents: Whether to group logs into incidents
        
    Returns:
        Tuple of (logs_df, incidents)
    """
    parser = LogParser()
    logs_df = parser.parse_log_file(log_file_path)
    
    incidents = None
    if group_incidents and len(logs_df) > 0:
        grouper = IncidentGrouper()
        incidents = grouper.group_by_error_bursts(logs_df)
        
        # Generate narratives
        for incident in incidents:
            incident['narrative'] = IncidentGrouper.create_incident_narrative(incident)
            incident['report'] = SyntheticReportGenerator.generate_report(incident, logs_df)
    
    return logs_df, incidents
