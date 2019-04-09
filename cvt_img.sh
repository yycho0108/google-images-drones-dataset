for d in ./data-png/*/; do
    pushd "$d"
    # jpg->jpg (in case for downloaded file amongst .jpg extension, encoding is incorrect)
    mogrify -format jpg -transparent-color white -background white -alpha background -flatten *.jpg
    # png->jpg (actual conversion)
    mogrify -format jpg -transparent-color white -background white -alpha background -flatten *.png
    #rm *.png
    popd
done
