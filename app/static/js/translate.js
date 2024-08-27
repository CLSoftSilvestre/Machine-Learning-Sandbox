class Translate{

    _language = "";

    datatable = [
        //General for all pages
        {key:"tr_home", pt:"Início", en:"Home", fr:"Commencer", default:"Home"},
        {key:"tr_about", pt:"Acerca", en:"About", fr:"À propos", default:"About"},
        {key:"tr_configuration", pt:"Configuração", en:"Configuration", fr:"Configuration", default:"Configuration"},
        {key:"tr_change_password", pt:"Alterar password", en:"Change password", fr:"Changer le mot de passefi", default:"Change password"},
        {key:"tr_logout", pt:"Desconectar", en:"Logout", fr:"Déconnexion", default:"Logout"},
        {key:"tr_warning", pt:"Atenção!", en:"Warning!", fr:"Avertissement!", default:"Warning!"},
        {key:"tr_success", pt:"Sucesso!", en:"Success!", fr:"Succès!", default:"Success!"},
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


        // Data Studio page
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

        // Buttons
        {key:"tr_cancel", pt:"Cancelar", en:"Cancel", fr:"Annuler", default:"Cancel"},
        {key:"tr_close", pt:"Fechar", en:"Close", fr:"Fermer", default:"Close"},
        {key:"tr_dismiss", pt:"Fechar", en:"Dismiss", fr:"Fermer", default:"Dismiss"},
        {key:"tr_save", pt:"Guardar", en:"Save", fr:"Sauvegarder", default:"Save"},


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