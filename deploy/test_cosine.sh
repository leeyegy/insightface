for img in 02.jpeg 03.jpg 04.png
do 
	python test_cosine.py --jiang_img $img | tee log/self_$img.txt
done
