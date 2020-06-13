import build_train_model

from getPredictArgs import getPredictArgs


def main():
    in_arg = getPredictArgs()
    
    input_path = in_arg.input
    checkpoint = in_arg.checkpoint
    topk = in_arg.top_k
    cat = in_arg.cat
    gpu = in_arg.gpu
    
    model =  build_train_model.load_checkpoin(path = checkpoint)
    flower_name, prob = build_train_model.predict(input_path, model, topk, power = gpu, category_names = cat)
    print(flower_name)
    print(prob)
    
    
    
if __name__ == "__main__":
    main()