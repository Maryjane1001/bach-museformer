printf '\n\n\n\n\n' | bash tgen/generation__mf-lmd6remi-x.sh $1 checkpoint_best.pt 17 | tee output_log/generation-$1.log
python tools/batch_extract_log.py output_log/generation-$1.log output/generation-$1 --start_idx 1
python tools/batch_generate_midis.py --encoding-method REMIGEN2 --input-dir output/generation-$1 --output-dir output/generation-$1
rm -rf checkpoints/mf-lmd6remi-$1/*0.pt
rm -rf checkpoints/mf-lmd6remi-$1/*5.pt
rm -rf checkpoints/mf-lmd6remi-$1/*last.pt
bash tval/val__mf-lmd6remi-x.sh $1 checkpoint_best.pt 1024 | grep 'valid on'
bash tval/val__mf-lmd6remi-x.sh $1 checkpoint_best.pt 10240 | grep 'valid on'