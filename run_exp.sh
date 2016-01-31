#!/bin/bash
#declare -a ds=("ItalyPowerDemand" "CBF" "FaceFour" "Gun_Point" "Coffee" "SonyAIBORobotSurface")
declare -a ds=("Lighting2" "MedicalImages" "Symbols" "Adiac" "SwedishLeaf" "FISH" "FacesUCR" "OSULeaf" "WordsSynonyms" "Cricket_X" "Cricket_Y" "Cricket_Z" "50words" "FaceAll" "ChlorineConcentration" "Haptics" "Two_Patterns" "MALLAT" "wafer" "CinC_ECG_torso" "yoga" "InlineSkate")
total=${#ds[*]}
declare -a type=(1 3 2)
#declare -a rMax=("24" "120" "340" "140" "280" "65")
jarName="shapelets.jar"

for ((i=0; i<=$(($total-1)); i++)); do
	for m in "${type[@]}"; do
		#java -jar $jarName --file "${ds[$i]}" --range 10 "${rMax[$i]}" --method $m | tee results/"${ds[$i]}"_"$m".log
		java -jar $jarName --file "${ds[$i]}" --range 10 0 --method $m | tee results/"${ds[$i]}"_"$m".log
		wait $!
		echo ""
	done
done

