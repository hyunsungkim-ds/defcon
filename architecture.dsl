workspace "DEFCON" "DEFensive CONtribution evaluator: a GNN-based framework for valuing defensive contributions in soccer by estimating Expected Possession Value changes." {

    model {
        researcher = person "Researcher" "Runs the training, evaluation, and analysis pipeline"

        defcon = softwareSystem "DEFCON" "Evaluates defensive contributions of soccer players using Graph Neural Networks and Expected Possession Value modeling" {
            preprocessor = container "Data Preprocessor" "Cleans tracking data, computes velocity and acceleration via Savitzky-Golay filter" "Python, datatools/preprocess.py"
            featureExtractor = container "Graph Feature Extractor" "Builds PyG graph representations from tracking snapshots with node/edge features" "Python, datatools/graph_feature.py"
            dataset = container "Action Dataset" "Loads graph features and labels, applies node dropping, edge sparsification, and inverse propensity weighting" "Python, dataset.py"
            gnnModels = container "GNN Models" "GAT/GCN/GIN encoder with task-specific MLP decoders for 6 component prediction tasks" "Python, PyTorch Geometric"
            trainer = container "Model Trainer" "Training loop with LR decay, checkpoint saving, and multi-task support" "Python, train.py"
            evaluator = container "Model Evaluator" "Runs trained models on test matches, reports metrics" "Python, test.py"
            xgModel = container "UxG Model" "Logistic regression for unblocked-shot expected goals, trained on Wyscout event data" "Python, scikit-learn"
            defconEngine = container "DEFCON Engine" "Estimates EPV components, assigns defensive credits per action, aggregates player scores" "Python, datatools/defcon.py"
            mainPipeline = container "Main Pipeline" "Orchestrates full-match inference: loads data, runs all models, computes player defensive scores" "Python, main.py"
            visualizer = container "Visualizer" "Pitch snapshots, score plots, heatmaps, pairwise credit matrices" "Python, matplotlib"
            notebook = container "Tutorial Notebook" "Interactive end-to-end match analysis and visualization workflow" "Jupyter, tutorial.ipynb"
        }

        trackingData = softwareSystem "Tracking Data" "Per-match Kloppy-format tracking Parquet files (velocity, position)" "External"
        eventData = softwareSystem "Event Data" "Per-match SPADL-format event Parquet files (synchronized with tracking)" "External"
        lineupData = softwareSystem "Lineup Data" "Match lineup Parquet file with player metadata" "External"
        wyscoutData = softwareSystem "Wyscout Open Dataset" "Public shot event CSV for UxG model training (CC-BY 4.0)" "External"
        savedWeights = softwareSystem "Saved Model Weights" "Trained GNN model checkpoints (.pt files)" "External"
        outputScores = softwareSystem "Player Scores" "Per-match defensive contribution scores (Parquet)" "External"

        researcher -> preprocessor "Preprocesses tracking data" "CLI"
        researcher -> featureExtractor "Extracts graph features" "CLI"
        researcher -> trainer "Trains GNN models" "CLI + shell scripts"
        researcher -> evaluator "Evaluates model performance" "CLI"
        researcher -> mainPipeline "Runs full inference" "CLI"
        researcher -> notebook "Analyzes matches interactively" "Jupyter"

        mainPipeline -> trackingData "Reads tracking frames" "pandas/pyarrow"
        mainPipeline -> eventData "Reads event actions" "pandas/pyarrow"
        mainPipeline -> lineupData "Reads player lineups" "pandas/pyarrow"
        mainPipeline -> defconEngine "Delegates component estimation to"
        mainPipeline -> outputScores "Writes player scores" "pyarrow"

        preprocessor -> trackingData "Reads raw tracking data" "pandas"

        featureExtractor -> trackingData "Reads processed tracking" "pandas"
        featureExtractor -> eventData "Reads event data" "pandas"

        dataset -> featureExtractor "Loads pre-built graph features" "torch.load (.pt)"

        trainer -> dataset "Loads training batches from"
        trainer -> gnnModels "Trains"
        trainer -> savedWeights "Saves best checkpoints to" ".pt files"

        evaluator -> dataset "Loads test batches from"
        evaluator -> gnnModels "Loads and evaluates"
        evaluator -> savedWeights "Loads weights from" "torch.load"

        defconEngine -> gnnModels "Runs inference on all 6 tasks via"
        defconEngine -> xgModel "Estimates unblocked-shot xG via"
        defconEngine -> savedWeights "Loads trained weights from" "torch.load"
        defconEngine -> mainPipeline "Returns player defensive scores"

        xgModel -> wyscoutData "Trains on shot events from" "pandas read_csv"

        notebook -> defconEngine "Runs component estimation"
        notebook -> visualizer "Generates plots and pitch diagrams"
    }

    views {
        systemContext defcon "SystemContext" {
            include *
            autoLayout
        }

        container defcon "Containers" {
            include *
            autoLayout
        }

        dynamic defcon "TrainingFlow" {
            researcher -> preprocessor "Preprocess raw tracking data"
            preprocessor -> trackingData "Read and clean tracking frames"
            researcher -> featureExtractor "Extract graph features"
            featureExtractor -> trackingData "Read processed tracking"
            featureExtractor -> eventData "Read synchronized events"
            researcher -> trainer "Train GNN models"
            trainer -> dataset "Load graph batches"
            trainer -> gnnModels "Forward pass and backprop"
            trainer -> savedWeights "Save best checkpoint"
            autoLayout
        }

        dynamic defcon "InferenceFlow" {
            researcher -> mainPipeline "Run full-match inference"
            mainPipeline -> trackingData "Load tracking data"
            mainPipeline -> eventData "Load event data"
            mainPipeline -> lineupData "Load lineup"
            mainPipeline -> defconEngine "Estimate all components"
            defconEngine -> savedWeights "Load trained GNN weights"
            defconEngine -> gnnModels "Run 6 GNN inference tasks"
            defconEngine -> xgModel "Estimate unblocked-shot xG"
            xgModel -> wyscoutData "Load shot training data"
            defconEngine -> mainPipeline "Return player defensive scores"
            mainPipeline -> outputScores "Write scores to Parquet"
            autoLayout
        }

        styles {
            element "Person" {
                shape Person
                background #08427B
                color #ffffff
            }
            element "Software System" {
                background #1168BD
                color #ffffff
            }
            element "External" {
                background #999999
                color #ffffff
            }
            element "Container" {
                background #438DD5
                color #ffffff
            }
            element "Component" {
                background #85BBF0
                color #000000
            }
        }
    }

}
