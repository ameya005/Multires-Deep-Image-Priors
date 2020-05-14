array=( 1 2 ) #5 7 8 9 )
for i in "${array[@]}"
do
    echo "Inpainting img $i"
    python3 main.py -i data/CelebA/00000$i.jpg -m data/CelebA/00000$i.png -a unet -o 128dim_unstable/img$i/gauss/ -lr 0.001 -e 3000 -s 10 -t gauss
    python3 main.py -i data/CelebA/00000$i.jpg -m data/CelebA/00000$i.png -a unet -o 128dim_unstable/img$i/lap/ -lr 0.001 -e 3000 -s 10 -t laplacian
done

    