ROOT_PATH=$1

PREFIX="generated_file_"
TEST_MODEL_NAME="mmhalsnowball"
KEY="original_answer"
DICT_PATH="./evaluation/data/mmhalsnowball_test.json"

echo "**************************************"
echo "    evaluating for MMHalSnowball      "
echo "**************************************"
python -m evaluation.eval \
    --prefix $PREFIX \
    --file-path $ROOT_PATH/$TEST_MODEL_NAME \
    --dict-path $DICT_PATH \
    --key $KEY

echo ""
echo ""

TEST_MODEL_NAME="wpi"
KEY="original_answer"
DICT_PATH="./evaluation/data/wpi_test.json"

echo "**************************************"
echo "         evaluating for WPI           "
echo "**************************************"
python -m evaluation.eval \
    --prefix $PREFIX \
    --file-path $ROOT_PATH/$TEST_MODEL_NAME \
    --dict-path $DICT_PATH \
    --key $KEY \
    --wpi-task
