import re
from datetime import datetime

LOG_PATTERN = re.compile(
    r'(?P<ip>\d+\.\d+\.\d+\.\d+) .* \[(?P<timestamp>[^\]]+)\] "(?P<request>[^"]*)" (?P<status>\d+) (?P<size>\d+|-)'
)

def parse_log_line(line):
    match = LOG_PATTERN.search(line)    
    if not match:
        return None
    data = match.groupdict()

    data['status'] = int(data['status'])
    data['size'] = int(data['size']) if data['size'] != '-' else 0

    try:
        data['timestamp'] = datetime.strptime(data['timestamp'], '%d/%b/%Y:%H:%M:%S %z')
    except ValueError:
        data['timestamp'] = None

    req_parts = data['request'].split()
    data['method'] = req_parts[0] if len(req_parts) > 0 else ''
    data['path'] = req_parts[1] if len(req_parts) > 1 else ''
    data['protocol'] = req_parts[2] if len(req_parts) > 2 else ''
    return data
