import configparser
class ConfigParser:

    # @staticmethod
    def read_config(config_name,section):
            try:
                config  = configparser.RawConfigParser()   
                configFilePath = 'app.config'
                config.read(configFilePath)
                rConfig = config[config_name][section]
            except Exception as error:
                print("Exception ocuured in Read Section {} {}".format(config_name,section))
                raise Exception("Not valid config name or section")
            return(rConfig)