# Classificador de Emails

## Descrição do Projeto

Este projeto é um classificador de emails que utiliza inteligência artificial para categorizar emails como **produtivos** ou **improdutivos**. O sistema oferece duas opções de classificação:

- **Modelo Treinado Local**: Utiliza um modelo de Machine Learning (Naive Bayes com TF-IDF) treinado localmente
- **Hugging Face API**: Utiliza um modelo avançado de linguagem (BART) via API do Hugging Face

## Estrutura do Projeto

```
email_classifier/
│
├── app.py                 # Interface principal do aplicativo (Streamlit)
├── model.py               # Modelo de classificação local (Naive Bayes)
├── hf_api.py              # Integração com Hugging Face API
├── emails.csv             # Base de dados de emails (atualmente vazia)
├── email_model.pkl        # Modelo treinado salvo (gerado automaticamente)
├── requirements.txt       # Dependências do projeto
├── document.md            # Esta documentação
└── __pycache__/           # Arquivos compilados Python
```

## Pré-requisitos

- Python 3.8 ou superior
- Ambiente virtual (recomendado)
- Conexão com internet (para Hugging Face API)

## Instalação e Configuração

### 1. Clonagem do Repositório

```bash
git clone https://github.com/matheusparreira22/email_classifier.git
cd email_classifier
```

### 2. Criação do Ambiente Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Instalação das Dependências

```bash
pip install -r requirements.txt
```

As dependências incluem:
- `streamlit`: Framework para interface web
- `scikit-learn`: Biblioteca de Machine Learning
- `pandas`: Manipulação de dados
- `numpy`: Computação numérica
- `requests`: Requisições HTTP para API

## Como Rodar o Projeto

### Método 1: Comando Direto (Recomendado)

```bash
streamlit run app.py
```

### Método 2: Usando Python Module

```bash
python -m streamlit run app.py
```

### Método 3: Caminho Completo (se houver problemas)

```bash
# No Windows
C:/caminho/para/venv/Scripts/python.exe -m streamlit run app.py

# No Linux/Mac
/caminho/para/venv/bin/python -m streamlit run app.py
```

Após executar qualquer um dos comandos acima, você verá uma mensagem como:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.1.100:8501
```

## Como Usar o Aplicativo

1. **Acesse o aplicativo**: Abra seu navegador e vá para `http://localhost:8501`

2. **Escolha o método de classificação**:
   - **Modelo Treinado (Local)**: Mais rápido, funciona offline
   - **Hugging Face API**: Mais preciso, requer internet

3. **Insira o conteúdo do email**:
   - Cole o texto do email na caixa de texto
   - Certifique-se de que há conteúdo antes de classificar

4. **Clique em "Classificar"**:
   - O sistema analisará o email
   - Mostrará o resultado: **PRODUTIVO** ou **IMPRODUTIVO**
   - Para Hugging Face, também mostra o nível de confiança

## Funcionalidades

### Classificação Automática
- Análise de conteúdo de emails
- Categorização em duas classes: produtivo/improdutivo
- Suporte a dois métodos de IA

### Treinamento Automático
- O modelo local é treinado automaticamente na primeira execução
- Dataset expandido com 50 exemplos balanceados
- Salvamento automático do modelo treinado

### Interface Intuitiva
- Interface web moderna com Streamlit
- Seleção fácil entre métodos de classificação
- Feedback visual dos resultados

## Exemplos de Uso

### Email Produtivo
```
Reunião de planejamento amanhã às 10h no escritório principal.
Por favor, confirme presença e prepare a apresentação do projeto.
```

**Resultado**: PRODUTIVO

### Email Improdutivo
```
PROMOÇÃO ESPECIAL! Compre agora e ganhe 50% de desconto!
Clique aqui para aproveitar esta oferta limitada!
```

**Resultado**: IMPRODUTIVO

## Notas Técnicas

### Modelo Local
- **Algoritmo**: Multinomial Naive Bayes
- **Vetorização**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Treinamento**: Automático com dataset interno
- **Arquivo**: `email_model.pkl`

### Hugging Face API
- **Modelo**: facebook/bart-large-mnli
- **Tarefa**: Zero-shot classification
- **Labels**: ["produtivo", "improdutivo"]
- **Token**: Configurado via variável de ambiente `HF_TOKEN`

### Segurança
- Token do Hugging Face configurado via variável de ambiente
- Modelo local não requer internet após treinamento inicial

## Solução de Problemas

### Erro: "streamlit command not found"
```bash
# Ative o ambiente virtual
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Instale novamente
pip install streamlit
```

### Erro: "No module named streamlit"
```bash
# Use o caminho completo do Python
python -m streamlit run app.py
```

### Erro na API do Hugging Face
- Verifique sua conexão com internet
- Configure a variável de ambiente `HF_TOKEN`
- A API pode ter limites de uso

### Modelo não carrega
- O modelo será treinado automaticamente na primeira execução
- Arquivo `email_model.pkl` será criado após o treinamento

## Desenvolvimento Futuro

- [ ] Adicionar mais categorias de classificação
- [ ] Implementar upload de arquivos CSV para treinamento personalizado
- [ ] Adicionar métricas de performance do modelo
- [ ] Criar API REST para integração com outros sistemas
- [ ] Melhorar interface com mais funcionalidades visuais

## Contribuição

Para contribuir com o projeto:

1. Faça um fork do repositório
2. Crie uma branch para sua feature
3. Faça commit das mudanças
4. Abra um Pull Request

## Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

---

**Última atualização**: Setembro 2025
**Versão**: 1.0.0