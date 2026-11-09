"""Single-image inference only; no training, dispatcher or experiment loops."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['embed', 'extract', 'trace'])
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--user-id', type=int, help='Numeric row in the frozen registry; required for embed/extract')
    parser.add_argument('--output', type=Path, required=True, help='PNG for embed, JSON for extract/trace')
    parser.add_argument('--gpu', choices=['1', '2', '3'], default='1')
    parser.add_argument('--pools', type=int, nargs='+', default=[10000, 100000, 1000000])
    args = parser.parse_args()
    if args.mode != 'trace' and args.user_id is None:
        parser.error('--user-id is required for embed/extract')
    if args.output.exists():
        parser.error('Output already exists; choose a new path')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    from watermark import Watermarker, load_registry, user_mapping
    config = json.loads(args.config.read_text(encoding='utf-8'))
    positions, transforms = load_registry(config)
    mapping = user_mapping(positions, transforms, args.user_id) if args.mode != 'trace' else None
    engine = Watermarker(config)
    image = engine.image(args.image)
    if args.mode == 'embed':
        engine.save_image(engine.embed(image, mapping), args.output)
    else:
        result = engine.extract(image, mapping) if args.mode == 'extract' else engine.trace(image, positions, transforms, args.pools)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
