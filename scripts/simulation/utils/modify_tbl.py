
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--update', action='append', nargs=4, help='ID MASS WIDTH LIFETIME')
    args = parser.parse_args()
    
    updates = {}
    if args.update:
        for u in args.update:
            updates[int(u[0])] = {'mass': float(u[1]), 'width': float(u[2]), 'lifetime': float(u[3])}
            
    with open(args.input) as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 6 and not line.strip().startswith('//') and not line.strip().startswith('#'):
            try:
                pid = int(parts[0])
                if pid in updates:
                    u = updates[pid]
                    # Preserve name and charge from original
                    name = parts[1]
                    chg = parts[2]
                    # ID(11) Name(24) Chg(3) Mass(11) Width(11) Life(13)
                    new_lines.append(f"{pid:>11} {name:<24} {chg:>3} {u['mass']:>11.5f} {u['width']:>11.5f} {u['lifetime']:>13.5E}\n")
                    continue
            except ValueError:
                pass
        new_lines.append(line)
        
    with open(args.output, 'w') as f:
        f.writelines(new_lines)

if __name__ == '__main__':
    main()

