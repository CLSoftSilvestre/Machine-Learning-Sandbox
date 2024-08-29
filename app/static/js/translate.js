class Translate{

    _language = "";

    datatable = [
        //General for all pages
        {key:"tr_home", pt:"Início", en:"Home", fr:"Commencer", default:"Home"},
        {key:"tr_about", pt:"Acerca", en:"About", fr:"À propos", default:"About"},
        {key:"tr_configuration", pt:"Configuração", en:"Configuration", fr:"Configuration", default:"Configuration"},
        {key:"tr_change_password", pt:"Alterar password", en:"Change password", fr:"Changer le mot de passefi", default:"Change password"},
        {key:"tr_logout", pt:"Desconectar", en:"Logout", fr:"Déconnexion", default:"Logout"},
        {key:"tr_really_logout", pt:"Deseja realmente desconectar-se?", en:"Do you really want to logout?", fr:"Voulez-vous vraiment vous déconnecter ?", default:"Do you really want to logout?"},
        {key:"tr_app_version", pt:"Versão app:", en:"App version:", fr:"Version de l'app:", default:"App version:"},
        {key:"tr_model_version", pt:"Versão modelo:", en:"Model version:", fr:"Version du modèle:", default:"Model version:"},
        {key:"tr_for_more_info", pt:"Para mais informações contatar:", en:"For more information please contact:", fr:"Pour plus d'informations, veuillez contacter:", default:"For more information please contact:"},
        {key:"tr_show_tutorials", pt:"Mostrar tutoriais", en:"Show tutorials", fr:"Afficher les tutoriels", default:"Show tutorials"},
        {key:"tr_username", pt:"Nome utilizador", en:"Username", fr:"Nom d'utilisateur", default:"Username"},
        {key:"tr_oldpassword", pt:"Palavra passe antiga", en:"Old password", fr:"Ancien mot de passe", default:"Old password"},
        {key:"tr_newpassword", pt:"Nova palavra passe", en:"New password", fr:"Nouveau mot de passe", default:"New password"},
        {key:"tr_repeatnewpassword", pt:"Repetir nova palavra passe", en:"Repeat new password", fr:"Répéter le nouveau mot de passe", default:"Repeat new password"},

            //Configurations
        {key:"tr_app_configuration", pt:"Configuração da aplicação", en:"Application configuration", fr:"Configuration des applications", default:"Application configuration"},
        {key:"tr_use_pygwalker", pt:"Utilizar a framework de análise de dados avançada PyGWalker", en:"Use PyGWalker advanced data visualization framework", fr:"Utiliser le framework avancé de visualisation de données PyGWalker", default:"Use PyGWalker advanced data visualization framework"},
        {key:"tr_pygwalker_info", pt:" A ativação da framework PyGWalker poderá impactar na performance do Estudio de Dados.", en:" Activating PyGWalker framework can have impact in the Data Studio performance.", fr:" L'activation du framework PyGWalker peut avoir un impact sur les performances de le Studio de données.", default:" Activating PyGWalker framework can have impact in the Data Studio performance."},
        {key:"tr_use_flows", pt:"Utilizar fluxos automatizados", en:"Use Automation Flows", fr:"Utiliser les flux d'automatisation", default:"Use Automation Flows"},
        {key:"tr_use_siemens", pt:"Utilizar conector de automação Siemens", en:"Use Siemens PLC connector", fr:"Utiliser le connecteur PLC Siemens", default:"Use Siemens PLC connector"},
        {key:"tr_use_mqtt", pt:"Utilizar conector MQTT", en:"Use MQTT connector", fr:"Utiliser le connecteur MQTT", default:"Use MQTT connector"},
        {key:"tr_use_opcua", pt:"Utilizar conector OPC UA", en:"Use OPC UA connector", fr:"Utiliser le connecteur OPC UA", default:"Use OPC UA connector"},
        {key:"tr_use_ble", pt:"Utilizar conector de Bluetooth Low Energy", en:"Use Bluetooth Low Energy connector", fr:"Utiliser le connecteur Bluetooth Low Energy", default:"Use Bluetooth Low Energy connector"},
        {key:"tr_ble_info", pt:" O conector BLE poderá não funcionar em contentores Docker.", en:" BLE connector may not work on Docker containers.", fr:" Le connecteur BLE peut ne pas fonctionner sur les conteneurs Docker.", default:" BLE connector may not work on Docker containers."},

        // Sandbox page
        {key:"tr_available_models", pt:"Todos os modelos disponiveis", en:"All available models", fr:"Tous les modèles disponibles", default:"All available models"},
        {key:"tr_sp_data_studio", pt:" Estúdio de dados", en:" Data studio", fr:" Studio de données", default:" Data studio"},
        {key:"tr_sp_import_model", pt:" Importar modelo existente", en:" Import existing model", fr:" Importer un modèle existant", default:" Import existing model"},
        {key:"tr_sp_refresh", pt:" Recarregar", en:" Refresh", fr:" Rafraîchir", default:" Refresh"},
        {key:"tr_sp_model_list", pt:" Lista de modelos", en:" Models list", fr:" Liste des modèles", default:" Models list"},

        {key:"tr_model_name", pt:"Nome do modelo", en:"Model name", fr:"Nom du modèle", default:"Model name"},
        {key:"tr_model_description", pt:"Descrição do modelo", en:"Model description", fr:"Description du modèle", default:"Model description"},
        {key:"tr_model_keywords", pt:"Palavras-chave", en:"Keywords", fr:"Mots-clés", default:"Keywords"},
        {key:"tr_model_train_datetime", pt:"Data e hora de treino", en:"Train datetime", fr:"Date et heure du train", default:"Train datetime"},
        {key:"tr_model_rmse", pt:"RMSE", en:"RMSE", fr:"RMSE", default:"RMSE"},
        {key:"tr_model_r2", pt:"R² / Precisão", en:"R² / Accuracy", fr:"R² / Précision", default:"R² / Accuracy"},
        {key:"tr_model_actions", pt:"Comandos", en:"Actions", fr:"Commandes", default:"Actions"},
        {key:"tr_not_defined", pt:"Não definido", en:"Not defined", fr:"Non défini", default:"Not defined"},
        {key:"tr_trained_older_version", pt:"Este modelo foi gerado por uma versão antiga do software. Algumas funcionalidades poderão não estar disponiveis.", en:"This model was trained in other software version. Some functionalities may not be available.", fr:"Ce modèle a été formé dans une autre version du logiciel. Certaines fonctionnalités peuvent ne pas être disponibles.", default:"This model was trained in other software version. Some functionalities may not be available."},

        // Data Studio page
        {key:"tr_data_studio", pt:"Estúdio de dados", en:"Data studio", fr:"Studio de données", default:"Data studio"},
        {key:"tr_load_dataset", pt:"Carregar dados", en:"Load dataset", fr:"Charger l'ensemble de données", default:"Load dataset"},
        {key:"tr_select_file", pt:"Selecionar ficheiro", en:"Select file", fr:"Sélectionner un fichier", default:"Select file"},
        {key:"tr_separator", pt:"Separador", en:"Separator", fr:"Séparateur", default:"Separator"},
        {key:"tr_decimal", pt:"Decimal", en:"Decimal", fr:"Décimal", default:"Decimal"},
        {key:"tr_parse_csv", pt:" Analisar CSV", en:" Parse CSV", fr:" Analyser le CSV", default:" Parse CSV"},
        {key:"tr_delete_dataset", pt:" Eliminar dados", en:" Delete dataset", fr:" Supprimer l'ensemble de données", default:" Delete dataset"},
        {key:"tr_clear_nulls", pt:" Eliminar nulos", en:" Clear nulls", fr:" Supprimer nulles", default:" Clear nulls"},
        {key:"tr_python_code", pt:" Código Python", en:" Python code", fr:" Code Python", default:" Python code"},
        {key:"tr_correlation_matrix", pt:" Matriz de correlação", en:" Correlation matrix", fr:" Matrice de corrélation", default:" Correlation matrix"},
        {key:"tr_scatter_plot", pt:" Nuvem de pontos", en:" Scatter plot", fr:" Nuage de points", default:" Scatter plot"},
        {key:"tr_data_exploration", pt:" Exploração de dados", en:" Data exploration", fr:" Exploration des données", default:" Data exploration"},
        {key:"tr_download_this_dataset", pt:" Descarregar conjunto de dados", en:" Download this dataset", fr:" Téléchargez données", default:" Download this dataset"},
        {key:"tr_use_for_ml", pt:" Utilizar dados para treino de algoritmo ML", en:" Use dataset for ML model training", fr:" Utiliser l' données pour la formation du modèle ML", default:" Use dataset for ML model training"},
        {key:"tr_data_statistics", pt:"Dados estatísticos", en:"Data statistics", fr:"Statistiques de données", default:"Data statistics"},
        {key:"tr_data_manipulator", pt:"Operações nos dados", en:"Data manipulator", fr:"Manipulation des données", default:"Data manipulator"},
        {key:"tr_boxplot_boards", pt:"Caixas de bigodes", en:"Boxplot boards", fr:"Tableaux de boîtes à moustaches", default:"Boxplot boards"},
        {key:"tr_data_operations", pt:"Histórico de operações", en:"Data operations", fr:"Opérations sur les données", default:"Data operations"},
        {key:"tr_variable", pt:"Variável", en:"Variable", fr:"Variable", default:"Variable"},
        {key:"tr_current_data_type", pt:"Tipo de dados atual", en:"Current data type", fr:"Type de données actuel", default:"Current data type"},
        {key:"tr_set_datatype", pt:"Definir tipo de dados", en:"Set dataype", fr:"Définir le type de données", default:"Set dataype"},
        {key:"tr_remove_column", pt:"Remover coluna", en:"Remove column", fr:"Supprimer la colonne", default:"Remove column"},
        {key:"tr_filter_column", pt:"Filtrar coluna", en:"Filter column", fr:"Filtre de colonne", default:"Filter column"},
        {key:"tr_remove_outliers", pt:"Remover valores anormais", en:"Remove outliers", fr:"Supprimer les valeurs aberrantes", default:"Remove outliers"},
        {key:"tr_set_data_name", pt:"Alterar nome da coluna", en:"Set data name", fr:"Définir le nom des données", default:"Set data name"},
        {key:"tr_msg_sure_change_type", pt:"Deseja realmente alterar o tipo de dados da coluna", en:"Are you sure you want to change the type of data of column", fr:"Êtes-vous sûr de vouloir modifier le type de données de la colonne", default:"Are you sure you want to change the type of data of column"},
        {key:"tr_msg_you_can_revert", pt:"Pode reverter esta modificação no separador Histórico de Operações.", en:"You can revert this action by deleting this operation in the Data Operations tab.", fr:"Vous pouvez annuler cette action en supprimant cette opération dans l'onglet Opérations sur les données.", default:"You can revert this action by deleting this operation in the Data Operations tab."},
        {key:"tr_new_data_type", pt:"Novo tipo de dados", en:"New data type", fr:"Nouveau type de données", default:"New data type"},
        {key:"tr_msg_sure_delete_column", pt:"Deseja realmente eliminar a coluna", en:"Are you sure you want to remove the column", fr:"Etes-vous sûr de vouloir supprimer la colonne", default:"Are you sure you want to remove the column"},
        {key:"tr_msg_filter_data_column", pt:"Filtrar dados da coluna", en:"Filter the data of column", fr:"Filtrer les données de la colonne", default:"Filter the data of column"},
        {key:"tr_operator", pt:"Operador", en:"Operator", fr:"Opérateur", default:"Operator"},
        {key:"tr_value", pt:"Valor", en:"Value", fr:"Valeur", default:"Value"},
        {key:"tr_msg_remove_outliers", pt:"Remover valores anormais da coluna", en:"Remove outliers from column", fr:"Supprimer les valeurs aberrantes de la colonne", default:"Remove outliers from column"},
        {key:"tr_upper_outlier", pt:"Remover valores anormais superiores", en:"Remove upper outliers", fr:"Supprimer les valeurs aberrantes supérieures", default:"Remove upper outliers"},
        {key:"tr_lower_outlier", pt:"Remover valores anormais inferiores", en:"Remove lower outliers", fr:"Supprimer les valeurs aberrantes inférieures", default:"Remove lower outliers"},
        {key:"tr_msg_change_name", pt:"Deseja realmente alterar o nome da coluna", en:"Are you sure you want to change the name of column", fr:"Etes-vous sûr de vouloir changer le nom de la colonne", default:"Are you sure you want to change the name of column"},
        {key:"tr_new_name", pt:"Novo nome", en:"New name", fr:"Nouveau nom", default:"New name"},

        {key:"tr_boxplot_outliers", pt:"caixa de bigodes e valores anormais", en:"boxplot and outliers", fr:"boîte à moustaches et valeurs aberrantes", default:"boxplot and outliers"},

        {key:"tr_pos", pt:"Pos", en:"Pos", fr:"Pos", default:"Pos"},
        {key:"tr_state", pt:"Estado", en:"State", fr:"État", default:"State"},
        {key:"tr_operation_type", pt:"Tipo de operação", en:"Operation type", fr:"Type d'opération", default:"Operation type"},
        {key:"tr_parameters", pt:"Parâmetros", en:"Parameters", fr:"Paramètres", default:"Parameters"},
        {key:"tr_actions", pt:"Comandos", en:"Actions", fr:"Commandes", default:"Actions"},

        {key:"tr_oper_setdatatype", pt:"Alterado o tipo de dados da coluna", en:"Set column new data type", fr:"Définir le nouveau type de données de la colonne", default:"Set column new data type"},
        {key:"tr_oper_remcol", pt:"Coluna removida", en:"Column removed", fr:"Colonne supprimée", default:"Column removed"},
        {key:"tr_oper_setcolname", pt:"Nome da coluna alterado", en:"Column name changed", fr:"Le nom de la colonne a changé", default:"Column name changed"},
        {key:"tr_oper_clearnull", pt:"Removidas linhas contendo nulos", en:"Delete rows with null data", fr:"Supprimer les lignes avec des données nulles", default:"Delete rows with null data"},
        {key:"tr_oper_filtercol", pt:"Filtro aplicado de acordo com dados de coluna", en:"Filter rows acording column values", fr:"Filtrer les lignes en fonction des valeurs des colonnes", default:"Filter rows acording column values"},
        {key:"tr_oper_remoutliers", pt:"Removidos dados anormais de coluna", en:"Removed column outliers", fr:"Valeurs aberrantes de colonnes supprimées", default:"Removed column outliers"},
        {key:"tr_oper_script", pt:"Script de python aplicado", en:"Python script applied", fr:"Script Python appliqué", default:"Python script applied"},

        {key:"tr_msg_remove_operation", pt:"Deseja realmente eliminar a operação", en:"Are you sure you want to remove the operation", fr:"Etes-vous sûr de vouloir supprimer l'opération", default:"Are you sure you want to remove the operation"},
        {key:"tr_msg_irreversible_action", pt:"Esta acção é irreversivel, contudo poderá voltar a aplicar na aba operações nos dados.", en:"This action is irreversible. However you can re-apply the data operation again.", fr:"Cette action est irréversible. Cependant, vous pouvez réappliquer l'opération de données.", default:"This action is irreversible. However you can re-apply the data operation again."},
        {key:"tr_oper_uuid", pt:" UUID operação:", en:" Operation UUID:", fr:" UUID d’opération :", default:" Operation UUID:"},
        {key:"tr_oper_type", pt:" Tipo de operação:", en:" Operation type:", fr:" Type d'opération:", default:" Operation type:"},
        {key:"tr_oper_params", pt:" Parâmetros da operação:", en:" Operation Parameters:", fr:" Paramètres d’opération:", default:" Operation Parameters:"},

        {key:"tr_edit_oper_pos", pt:"Editar a operação na posição", en:"Edit operation position", fr:"Modifier la position d'opération", default:"Edit operation position"},
        {key:"tr_editing_oper", pt:"Editando a operação", en:"Editing operation", fr:"Opération d'édition", default:"Editing operation"},
        {key:"tr_old_params", pt:"Parâmetros antigos", en:"Old parameters", fr:"Anciens paramètres", default:"Old parameters"},
        {key:"tr_dataset_preview", pt:" Pré-visualização do conjunto de dados (primeiros 10 registos)", en:" Dataset preview (only first 10 records)", fr:" Aperçu de l'ensemble de données (uniquement les 10 premiers enregistrements)", default:" Dataset preview (only first 10 records)"},
        {key:"tr_var_correlation_matrix", pt:"Matrix de correlação das variáveis", en:"Variables correlation matrix", fr:"Matrice de corrélation des variables", default:"Variables correlation matrix"},

        // Python code Modal
        {key:"tr_apply_python_script_oper", pt:"Aplicar operação de dados em código Python", en:"Apply a Python script data operation", fr:"Appliquer une opération de données de script Python", default:"Apply a Python script data operation"},
        {key:"tr_script_code", pt:"Script de código:", en:"Script code:", fr:"Code du script:", default:"Script code:"},
        {key:"tr_available_var_func", pt:"Variáveis e funções disponíveis", en:"Available variables and functions", fr:"Variables et fonctions disponibles", default:"Available variables and functions"},
        {key:"tr_data_variables", pt:"Variáveis de dados", en:"Data variables", fr:"Variables de données", default:"Data variables"},
        {key:"tr_custom_functions", pt:"Funções específicas", en:"Custom functions", fr:"Fonctions personnalisées", default:"Custom functions"},
        {key:"tr_usage_examples", pt:"Alguns exemplos de utilização:", en:"Some usage examples:", fr:"Quelques exemples d'utilisation :", default:"Some usage examples:"},
        {key:"tr_msg_code_not_validated", pt:"O código não é validado antes da execução. Comportamentos estranhos podem acontecer devido a erros ortográficos. Use com moderação.", en:"The code is not validated before executed. Strange behaviours may occur due to misspelling. Python is case-sensitive. Please use it carefully.", fr:"Le code n'est pas validé avant d'être exécuté. Des comportements étranges peuvent survenir en raison d'une faute d'orthographe. Python est sensible à la casse. Veuillez l'utiliser avec précaution.", default:"The code is not validated before executed. Strange behaviours may occur due to misspelling. Python is case-sensitive. Please use it carefully."},

        {key:"tr_func_add_column", pt:"Adicionar coluna ao dataset", en:"Add column to dataset", fr:"Ajouter une colonne à l'ensemble de données", default:"Add column to dataset"},
        {key:"tr_func_agg_week", pt:"Agregar dados por semana", en:"Aggregate per week", fr:"Total par semaine", default:"Aggregate per week"},
        {key:"tr_func_agg_month", pt:"Agregar dados por mês", en:"Aggregate per month", fr:"Total par mois", default:"Aggregate per month"},
        {key:"tr_func_expand_cat", pt:"Expandir categorias para colunas", en:"Expand categories to columns", fr:"Étendre les catégories aux colonnes", default:"Expand categories to columns"},

        {key:"tr_func_rep_nan_value", pt:"Substituir valores não numéricos por valor", en:"Replace NaN with number", fr:"Remplacez pas une donnée numérique par une valeur", default:"Replace NaN with number"},
        {key:"tr_func_rep_nan_avg", pt:"Substituir valores não numéricos por valor médio", en:"Replace NaN with average", fr:"Remplacez pas une donnée numérique par une valeur moyenne", default:"Replace NaN with average"},
        {key:"tr_func_rep_nan_med", pt:"Substituir valores não numéricos por valor da médiana", en:"Replace NaN with median", fr:"Remplacez pas une donnée numérique par une valeur médian", default:"Replace NaN with median"},
        {key:"tr_func_rep_nan_mode", pt:"Substituir valores não numéricos por valor da moda", en:"Replace NaN with mode", fr:"Remplacez pas une donnée numérique par une valeur mode", default:"Replace NaN with mode"},
        {key:"tr_func_rem_duplicated", pt:"Remover linhas com valores repetidos", en:"Remove duplicated entries", fr:"Supprimer les entrées en double", default:"Remove duplicated entries"},

        {key:"tr_ex_copy_col", pt:"Criar cópia de coluna", en:"Copy a specific column", fr:"Copier une colonne spécifique", default:"Copy a specific column"},
        {key:"tr_ex_copy_col_desc", pt:"Para criar uma coluna baseada em outra utilize a seguinte função", en:"To create a new column based on other just perform the procedure", fr:"Pour créer une nouvelle colonne basée sur une autre, effectuez simplement la procédure", default:"To create a new column based on other just perform the procedure"},
        {key:"tr_ex_copy_col_calc", pt:"Criar uma coluna com valores calculados", en:"Create new column with calculated values", fr:"Créer une nouvelle colonne avec des valeurs calculées", default:"Create new column with calculated values"},
        {key:"tr_ex_copy_col_calc_desc", pt:"Para criar uma coluna com valores calculados utilize a seguinte sintaxe", en:"To add a new calculated column use the following syntax", fr:"Pour ajouter une nouvelle colonne calculée, utilisez la syntaxe suivante", default:"To add a new calculated column use the following syntax"},
        {key:"tr_ex_week_from_date", pt:"Criar uma coluna com numero da semana a partir de data", en:"Create week number from date", fr:"Créer un numéro de semaine à partir de la date", default:"Create week number from date"},
        {key:"tr_ex_week_from_date_desc", pt:"Primeiro confirme que a coluna de data está formatada com o tipo Data e Hora. Para criar a coluna com o número da semana utilize a seguinte função", en:"Ensure first that the date column is set with datetime format. To add a new calculated column with the week number from date use the following syntax", fr:"Assurez-vous d’abord que la colonne de date est définie au format datetime. Pour ajouter une nouvelle colonne calculée avec le numéro de semaine à partir de la date, utilisez la syntaxe suivante", default:"Ensure first that the date column is set with datetime format. To add a new calculated column with the week number from date use the following syntax"},
        {key:"tr_ex_encode_features", pt:"Codificar dados categoricos", en:"Encode categorical features", fr:"Encoder les caractéristiques catégorielles", default:"Encode categorical features"},
        {key:"tr_ex_encode_features_desc", pt:"Esta função codifica recursos categóricos. Por exemplo, transforma ['baixo', 'médio', 'alto', 'médio', 'baixo'] em [0, 1, 2, 1, 0]. Para transformar os dados aplique o seguinte código", en:"This function encodes categorical features. For example it transforms ['low', 'mid', 'high', 'mid', 'low'] into [0, 1, 2, 1, 0]. To transform the data apply the following code", fr:"Cette fonction code les caractéristiques catégorielles. Par exemple, il transforme ['bas', 'moyen', 'haut', 'moyen', 'bas'] en [0, 1, 2, 1, 0]. Pour transformer les données, appliquez le code suivant", default:"This function encodes categorical features. For example it transforms ['low', 'mid', 'high', 'mid', 'low'] into [0, 1, 2, 1, 0]. To transform the data apply the following code"},
        
        {key:"tr_console", pt:"Consola", en:"Console", fr:"Console", default:"Console"},
        {key:"tr_terminal_output", pt:"Saída do terminal do estúdio de dados", en:"Data studio terminal output", fr:"Sortie du terminal de studio de données", default:"Data studio terminal output"},

        // Train page
        {key:"tr_train_new_model", pt:"Treinar novo modelo", en:"Train new model", fr:"Former un nouveau modèle", default:"Train new model"},
        {key:"tr_back_data_studio", pt:"Regressar ao Estúdio de Dados", en:"Back to Data Studio", fr:"Retour à Studio de Données", default:"Back to Data Studio"},
        {key:"tr_data_properties", pt:"Propriedades dos dados", en:"Data properties", fr:"Propriétés des données", default:"Data properties"},
        {key:"tr_variable_selector", pt:"Selecção de variável", en:"Variable selector", fr:"Sélecteur de variables", default:"Variable selector"},
        {key:"tr_model_selector", pt:"Selecção de modelo", en:"Model selector", fr:"Sélecteur de modèle", default:"Model selector"},
        {key:"tr_current_unit", pt:"Unidade atual", en:"Current unit", fr:"Unité actuelle", default:"Current unit"},
        {key:"tr_new_unit", pt:"Nova unidade", en:"New unit", fr:"Nouvelle unité", default:"New unit"},
        {key:"tr_set_y", pt:"Definir como coluna Y (dependente)", en:"Set as Y column (dependent)", fr:"Définir comme colonne Y (dépendant)", default:"Set as Y column (dependent)"},
        {key:"tr_reg_models", pt:"Modelos de regressão", en:"Regression models", fr:"Modèles de régression", default:"Regression models"},
        {key:"tr_reg_models_desc", pt:"Selecione um destes modelos quando o valor a prever é continuo.", en:"Select one of those models when the value to predict is as continuous value.", fr:"Sélectionnez l'un de ces modèles lorsque la valeur à prédire est une valeur continue.", default:"Select one of those models when the value to predict is as continuous value."},

        {key:"tr_lin_reg", pt:"Regressão linear", en:"Linear regression", fr:"Régression linéaire", default:"Linear regression"},
        {key:"tr_lin_reg_desc", pt:"Um modelo de regressão linear é um algoritmo de aprendizado supervisionado simples e amplamente utilizado para resolver tarefas de regressão. Funciona encontrando a relação linear entre os recursos de entrada e a variável de destino.", en:"A Linear Regression model is a simple and widely used supervised learning algorithm for solving regression tasks. It works by finding the linear relationship between the input features and the target variable.", fr:"Un modèle de régression linéaire est un algorithme d'apprentissage supervisé simple et largement utilisé pour résoudre des tâches de régression. Cela fonctionne en trouvant la relation linéaire entre les caractéristiques d'entrée et la variable cible.", default:"A Linear Regression model is a simple and widely used supervised learning algorithm for solving regression tasks. It works by finding the linear relationship between the input features and the target variable."},

        {key:"tr_knn_reg", pt:"Regressão kNN", en:"kNN regression", fr:"Régression kNN", default:"kNN regression"},
        {key:"tr_knn_reg_desc", pt:"Um modelo de regressão K-Nearest Neighbors (KNN) é um algoritmo de aprendizagem supervisionado usado para tarefas de regressão. Ele funciona usando os K pontos de dados mais próximos para fazer previsões sobre a variável alvo.", en:"A K-Nearest Neighbors (KNN) regression model is a supervised learning algorithm used for regression tasks. It works by using the K closest data points to make predictions about the target variable.", fr:"Un modèle de régression K-Nearest Neighbours (KNN) est un algorithme d'apprentissage supervisé utilisé pour les tâches de régression. Il fonctionne en utilisant les K points de données les plus proches pour faire des prédictions sur la variable cible.", default:"A K-Nearest Neighbors (KNN) regression model is a supervised learning algorithm used for regression tasks. It works by using the K closest data points to make predictions about the target variable."},


        {key:"tr_class_models", pt:"Modelos de classificação", en:"Classification models", fr:"Modèles de classification", default:"Classification models"},
        {key:"tr_class_models_desc", pt:"Selecione um destes modelos quando o valor a prever é uma categoria.", en:"Select one of those models when the value to predict is as categorical value.", fr:"Sélectionnez l'un de ces modèles lorsque la valeur à prédire est une valeur catégorielle.", default:"Select one of those models when the value to predict is as categorical value."},


        {key:"tr_anom_models", pt:"Modelos de detecção de anomalias", en:"Anomaly detection models", fr:"Modèles de détection d'anomalies", default:"Anomaly detection models"},
        {key:"tr_anom_models_desc", pt:"Selecione um destes modelos quando o valor a prever é uma categoria.", en:"Select one of those models when is needed to detect anomaly in the data. Anomaly detection don't require variable Y selection.", fr:"Sélectionnez l'un de ces modèles lorsque cela est nécessaire pour détecter une anomalie dans les données. La détection des anomalies ne nécessite pas de sélection de variable Y", default:"Select one of those models when is needed to detect anomaly in the data. Annomaly detection don't require variable Y selection."},


        // States
        {key:"tr_error", pt:"Erro!", en:"Error!", fr:"Erreur!", default:"Error!"},
        {key:"tr_warning", pt:"Atenção!", en:"Warning!", fr:"Avertissement!", default:"Warning!"},
        {key:"tr_success", pt:"Sucesso!", en:"Success!", fr:"Succès!", default:"Success!"},

        // Buttons
        {key:"tr_cancel", pt:"Cancelar", en:"Cancel", fr:"Annuler", default:"Cancel"},
        {key:"tr_close", pt:"Fechar", en:"Close", fr:"Fermer", default:"Close"},
        {key:"tr_dismiss", pt:"Fechar", en:"Dismiss", fr:"Fermer", default:"Dismiss"},
        {key:"tr_save", pt:"Guardar", en:"Save", fr:"Sauvegarder", default:"Save"},
        {key:"tr_update", pt:" Atualizar", en:" Update", fr:" Mise à jour", default:"Update"},
        {key:"tr_update_units", pt:" Atualizar unidades", en:" Update units", fr:" Mettre à jour les unités", default:"Update units"},
        {key:"tr_use_this_model", pt:"Utilizar este modelo", en:"Use this model", fr:"Utilisez ce modèle", default:"Use this model"},

        // Data types
        {key:"tr_datetime", pt:"Data e Hora", en:"Datetime", fr:"Dateheure", default:"Datetime"},
        {key:"tr_integer", pt:"Inteiro", en:"Integer", fr:"Entier", default:"Integer"},
        {key:"tr_float", pt:"Decimal", en:"Float", fr:"Décimal", default:"Float"},

        {key:"tr_empty", pt:" ", en:" ", fr:" ", default:" "},
    ]

    constructor(lg){
        //Selected language
        this._language = lg;
    }

    getTranslation(key, lg){
        switch (lg) {
            case "EN":
                try {
                    return this.datatable.find(x => x.key === key).en;
                } catch (error) {
                    return null;
                }
                break;
            case "PT":
                try {
                    return this.datatable.find(x => x.key === key).pt;
                } catch (error) {
                    return null;
                }
                break;
            case "FR":
                try {
                    return this.datatable.find(x => x.key === key).fr;
                } catch (error) {
                    return null;
                }
                break;
            default:
                return this.datatable.find(x => x.key === key).default;
                break;
        }   
    }
}