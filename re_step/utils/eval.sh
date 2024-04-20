for metric in p@1 map err@5
do
  java -jar ./RankLib-2.16.jar -test "sorted_test_best_model.pt.dat" \
  -metric2T $metric -idv "${metric}_best_model.pt.txt"
done