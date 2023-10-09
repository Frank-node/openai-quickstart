import sys
import os
import gradio as gr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import ArgumentParser, ConfigLoader, LOG
from translator import PDFTranslator
from model import GLMModel, OpenAIModel

def translation(input_file, source_language, target_language, file_format):
    LOG.debug(f"[翻译任务]\n源文件: {input_file.name}\n源语言: {source_language}\n目标语言: {target_language}")

    output_file_path = Translator.translate_pdf(
        input_file.name, file_format,
        source_language=source_language,
        target_language=target_language)
    #LOG.info('output_file_path: ' + output_file_path)
    return output_file_path

def launch_gradio():

    iface = gr.Interface(
        fn=translation,
        title="OpenAI-Translator v2.0",
        inputs=[
            gr.File(label="上传PDF文件"),
            gr.Dropdown(['English','Chinese','French', 'Spainish','Portuguese', ],label="源语言（默认：English）", value="English"),
            gr.Dropdown(['Chinese', 'English', 'French', 'Spainish','Portuguese'],label="目标语言（默认：Chinese）", value="Chinese"),
            gr.Dropdown(['markdown','pdf'], label="翻译文件格式（默认：markdown）", value="markdown")
        ],
        outputs=[
            gr.File(label="下载翻译文件")
        ],
        allow_flagging="never"
    )

    iface.launch(share=True, server_name="0.0.0.0")

def initialize_translator():
    # 解析命令行
    argument_parser = ArgumentParser()
    args = argument_parser.parse_arguments()

    # 初始化配置单例
    config_loader = ConfigLoader(args.config)

    config = config_loader.load_config()  
    model_name = args.openai_model if args.openai_model else config['OpenAIModel']['model']
    api_key = args.openai_api_key if args.openai_api_key else os.environ['OPENAI_API_KEY'] #config['OpenAIModel']['api_key']
    # file_format = args.file_format if args.file_format else config['common']['file_format']

    model = OpenAIModel(model=model_name, api_key=api_key)
    # 实例化 PDFTranslator 类，并调用 translate_pdf() 方法
    global Translator
    Translator = PDFTranslator(model)


if __name__ == "__main__":
    # 初始化 translator
    initialize_translator()
    # 启动 Gradio 服务
    launch_gradio()
