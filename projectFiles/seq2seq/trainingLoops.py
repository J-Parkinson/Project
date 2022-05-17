import torch

from projectFiles.seq2seq.devTestFunction import validationMultipleBatches
from projectFiles.seq2seq.trainingFunction import train


def trainOneEpoch(batches, batchSize, batchesBetweenTrainingPlots, batchesBetweenValidationCheck,
                  curriculumLearningSpec,
                  datasetName,
                  decoder, decoderOptimizer, decoderNoLayers,
                  encoder, encoderOptimizer,
                  epochNo,
                  fileSaveDir,
                  gradientClip,
                  iterationGlobal,
                  minDevLoss, minLossAverage,
                  noEpochs,
                  plotDevLosses, plotLosses,
                  resultsGlobal,
                  teacherForcingRatio,
                  timer):
    # Load batches for each iteration
    trainingBatches = batches.trainDL
    validationBatches = batches.devDL

    lossTotal = 0  # Reset every batchesBetweenValidationCheck

    # Initializations
    print('Initializing ...')

    # Training loop
    print("Training...")

    for iteration, batch in enumerate(trainingBatches, 1):
        iterationGlobal += 1
        # Extract fields from batch
        inputIndices = batch["inputEmbeddings"]
        outputIndices = batch["outputEmbeddings"]
        lengths = batch["lengths"]
        maxSimplifiedLength = batch["maxSimplifiedLength"]
        mask = batch["mask"]

        # Run a training iteration with batch
        loss = train(inputIndices, lengths, outputIndices, mask, maxSimplifiedLength, encoder, decoder, decoderNoLayers,
                     encoderOptimizer, decoderOptimizer, batchSize, gradientClip, teacherForcingRatio)

        print(f"Iteration: {iteration}; "
              f"Iteration global: {iterationGlobal}; "
              f"Percent complete: {iteration / len(trainingBatches) * 100}%; "
              f"Average loss: {loss}")

        lossTotal += loss

        if iteration % batchesBetweenTrainingPlots == 1:
            lossAverage = lossTotal / min(batchesBetweenValidationCheck, iteration)
            minLossAverage = lossAverage if not minLossAverage else min(minLossAverage, lossAverage)
            plotLosses[(iterationGlobal, iteration, epochNo)] = lossAverage
            print(f"Batch: {iteration}; Loss average: {lossAverage}")
            lossTotal = 0

        if iteration % batchesBetweenValidationCheck == 1:
            print("VALIDATION CALCULATION___________________________________________________________________")
            encoder.eval()
            decoder.eval()

            # Calculate validation loss
            devLoss, results = validationMultipleBatches(validationBatches, encoder, decoder,
                                                         decoderNoLayers, batchSize)
            minDevLoss = devLoss if not minDevLoss else min(minDevLoss, devLoss)

            plotDevLosses[(iterationGlobal, iteration, epochNo)] = devLoss
            resultsGlobal[(iterationGlobal, iteration, epochNo)] = results

            if minDevLoss == devLoss:
                optimalEncoder = encoder.state_dict()
                optimalDecoder = decoder.state_dict()
                with open(f"{fileSaveDir}/epochRun.txt", "w+") as file:
                    file.write(
                        f"Epoch {epochNo}---------------\n"
                        f"Batch no in epoch:             {iteration}/{len(trainingBatches)}\n"
                        f"Batch no across epochs:        {iterationGlobal}/{len(trainingBatches) * noEpochs}\n"
                        f"Dataset name:                  {datasetName}\n"
                        f"Curriculum learning for epoch: {curriculumLearningSpec.flag.name}\n"
                        f"Time ran:                      {timer.checkTimeDiff()}\n"
                        f"Min loss average:              {minLossAverage}\n"
                        f"Min dev loss:                  {minDevLoss}\n"
                    )
                torch.save(optimalEncoder, f"{fileSaveDir}/encoder.pt")
                torch.save(optimalDecoder, f"{fileSaveDir}/decoder.pt")

            print("_____________________")
            print(f"Batch {iteration}")
            timer.printTimeDiff()
            timer.printTimeBetweenChecks()
            print(f"Min avg training loss per {batchesBetweenValidationCheck} iterations: "
                  f"{minLossAverage / batchesBetweenValidationCheck}")
            print(f"Validation loss: {devLoss}")

            encoder.train()
            decoder.train()

            print("__________________________________________________________________________________")

    return decoder, decoderOptimizer, encoder, encoderOptimizer, iterationGlobal, minLossAverage, minDevLoss, plotDevLosses, plotLosses, resultsGlobal, timer


def trainMultipleEpochs(batches, batchSize, batchesBetweenTrainingPlots, batchesBetweenValidationCheck,
                        curriculumLearningAfterFirstEpoch, curriculumLearningSpec,
                        datasetName,
                        decoder, decoderOptimizer, decoderNoLayers,
                        encoder, encoderOptimizer,
                        fileSaveDir,
                        gradientClip,
                        noEpochs,
                        teacherForcingRatio,
                        timer):
    iterationGlobal = 0
    plotLosses = {}
    plotDevLosses = {}
    resultsGlobal = {}
    minDevLoss = None
    minLossAverage = None

    for epochNo in range(1, noEpochs + 1):
        if epochNo == 2:
            curriculumLearningSpec = curriculumLearningAfterFirstEpoch
        print(f"EPOCH {epochNo}")
        decoder, decoderOptimizer, encoder, encoderOptimizer, iterationGlobal, minLossAverage, minDevLoss, \
        plotDevLosses, plotLosses, resultsGlobal, timer = trainOneEpoch(batches, batchSize,
                                                                        batchesBetweenTrainingPlots,
                                                                        batchesBetweenValidationCheck,
                                                                        curriculumLearningSpec,
                                                                        datasetName,
                                                                        decoder, decoderOptimizer, decoderNoLayers,
                                                                        encoder, encoderOptimizer,
                                                                        epochNo,
                                                                        fileSaveDir,
                                                                        gradientClip,
                                                                        iterationGlobal,
                                                                        minDevLoss, minLossAverage,
                                                                        noEpochs,
                                                                        plotDevLosses, plotLosses,
                                                                        resultsGlobal,
                                                                        teacherForcingRatio,
                                                                        timer)
    return decoder, encoder, iterationGlobal, plotDevLosses, plotLosses, resultsGlobal
