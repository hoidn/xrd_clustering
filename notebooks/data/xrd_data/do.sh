for file in *.csv; do
  # Remove Windows-style carriage returns
  dos2unix "$file"
done
