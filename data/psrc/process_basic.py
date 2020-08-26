# after exporting to csv
csv_file = 'psrc_data.csv'
with open(csv_file, 'r') as f:
    lines = f.read()
with open('psrc_data_processed.csv', 'w') as f:
    lines = lines.replace('Unchecked', '0')
    lines = lines.replace('Checked', '1')
    lines = lines.replace('Yes', '1')
    lines = lines.replace('No', '0')
    lines = lines.replace('Unknown/not reported', '')
    f.write(lines)
