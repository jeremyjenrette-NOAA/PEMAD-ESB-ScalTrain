DEST="../data/images/2023tr/"
MANIFEST="../data/raw/sources_2023tr.txt"

mkdir -p "$DEST"

# -P runs multiple copies at once (tune 4/8/16)
# cat "$MANIFEST" | xargs -I{} -P 2 cp -n "{}" "$DEST"
# rsync -a --ignore-existing --info=progress2 --files-from="$MANIFEST" / "$DEST"
cat "$MANIFEST" | xargs -n 1 -P 2 -I{} cp -n "{}" "$DEST/"