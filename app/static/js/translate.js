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


        // States
        {key:"tr_error", pt:"Erro!", en:"Error!", fr:"Erreur!", default:"Error!"},
        {key:"tr_warning", pt:"Atenção!", en:"Warning!", fr:"Avertissement!", default:"Warning!"},
        {key:"tr_success", pt:"Sucesso!", en:"Success!", fr:"Succès!", default:"Success!"},

        // Buttons
        {key:"tr_cancel", pt:"Cancelar", en:"Cancel", fr:"Annuler", default:"Cancel"},
        {key:"tr_close", pt:"Fechar", en:"Close", fr:"Fermer", default:"Close"},
        {key:"tr_dismiss", pt:"Fechar", en:"Dismiss", fr:"Fermer", default:"Dismiss"},
        {key:"tr_save", pt:"Guardar", en:"Save", fr:"Sauvegarder", default:"Save"},

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