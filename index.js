import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs){
    const model = tf.sequential()

    // Primeira camada da rede:
    // Entrada com 7 posições (idade normalizada + 3 cores + 3 localizações)

    // 80 neurônios porque tem pouca base de treino
    // quanto mais neurônios mais complexidade a rede pode aprender
    // consequentemente mais processamento ela vau usar

    // A ativação ReLU age como um filtro
    // É como se ela deixasse somente os dados interessantes seguirem viagem na rede
    // Se a informação que chegou nesse neurônio é positiva, segue adiante!
    // Se for zero ou negativa, pode jogar fora, não serve para nada
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }))

    // Saída tem 3 neurônios
    // Um para cada categoria (premium, medium, basic)

    // activation: softmax -> normaliza a saída em probabilidades
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

    // Compilando o modelo
    // optimizer Adam (Adaptive Moment Estimation)
    // é um treinador pessoal moderno para redes neurais
    // ele ajusta os pesos de forma eficiente e inteligente
    // aprende com o histórico de erros e acertos

    // loss: categoricalCrossentropy
    // Ele compara o que o modelo "acha" (os scores de cada categoria) 
    // com a reposta certa
    // a categoria premium será sempre [1, 0, 0]

    // Quanto mais distante a previsão do modelo estiver da resposta correta
    // maior o erro (loss)
    // Exemplo clássico: classificação de imagens, sistemas de recomendação,
    // categorização de usuário
    // qualquer coisa em que a resposta certa é "apenas uma entre várias possíveis"

    model.compile({ 
        optimizer: 'adam', 
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })

    // Treinamento do modelo
    // verbose: desabilita o log interno (e usa só o callback)
    // epochs: quantidade de vezes que o modelo vai passar pelo dataset no treino
    // shuffle: embaralha os dados a cada iteração (ou "epoch") para evitar qualquer viés 
    // (ou "BIAS"), ou seja, para não viciar o modelo e não causar "overfitting" 
    // que é o que ocorre quando o modelo aprende demais sobre o dataset de treino e não consegue
    // generalizar para novos dados
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,
            epochs: 100,
            shuffle: true,
            callbacks: {
                //onEpochEnd: (epoch, logs) => console.log(`Epoch ${epoch}: loss = ${logs.loss}`
                //)
            }
        }
    )

    return model
}


async function predict(model, pessoa) {
    // Tranforma o array JS para um tensor 2D (tfjs)
    const tfInput = tf.tensor2d(pessoa)
    
    // Faz a predição (output será um vetor de 3 probabilidades)
    const pred = model.predict(tfInput)
    const predArray = await pred.array()    
    return predArray[0].map((prob, index) => ({ prob, index }))
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// quanto mais dados melhor
// assim o algoritmo consegue aprender melhor padrões complexos presentes nos dados
const model = await trainModel(inputXs, outputYs)

const pessoa = { nome: 'zé', idade: 28, cor: 'verde', localizacao: 'Curitiba'}

// Normalizando a idade da nova pessoa usando o mesmo padrão do treino
// Exemplo: idade_min = 25, idade_max = 40, idade_norm = (28 - 25) / (40 - 25) = 0.2
// const idadeNormalizada = (pessoa.idade - 25) / (40 - 25) // idade normalizada entre 0 e 1


const pessoaTensorNormalizado = [
    [
        0.2, // idade normalizada
        1,   // azul
        0,   // vermelho
        0,   // verde
        0,   // localização: São Paulo
        1,   // localização: Rio
        0    // localização: Curitiba
    ]
]

const predictions = await predict(model, pessoaTensorNormalizado)

const results = predictions.sort((a, b) => b.prob - a.prob).map(p => `${labelsNomes[p.index]} (${(p.prob*100).toFixed(2)}%)`).join('\n')

console.log(results)
