def xainano_sequence_generator(generator, config, parser, batch_size):
    #def sequence_generator():
    while True:
        inputs = []
        targets = []
        for index in range(batch_size):
            tokens = []
            generator.generate_formula(tokens, config)
            image = parser.parse(tokens)
            tokens.append("<end>")
            inputs.append(image)
            targets.append(tokens)
        yield (inputs, targets)
    #return sequence_generator
