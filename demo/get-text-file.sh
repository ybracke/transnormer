mkdir -p ./data && wget -nv -P ./data "https://www.deutschestextarchiv.de/media/download/dtak/2020-10-23/original/1700-1799.zip"
unzip  -q ./data/1700-1799.zip -d ./data
rm ./data/1700-1799.zip
find ./data/1700-1799 -type f ! -name "robins_artillerie*" -delete
