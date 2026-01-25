<?php 

if ( ! defined( 'ABSPATH' ) ) exit; // Exit if accessed directly

use Google\Cloud\TextToSpeech\V1\AudioConfig;
use Google\Cloud\TextToSpeech\V1\AudioEncoding;
use Google\Cloud\TextToSpeech\V1\SynthesisInput;
use Google\Cloud\TextToSpeech\V1\TextToSpeechClient;
use Google\Cloud\TextToSpeech\V1\VoiceSelectionParams;
use Google\Cloud\TextToSpeech\V1\SsmlVoiceGender;

#[AllowDynamicProperties]
class MagicAI_GoogleTTS {

    /**
	 * Instance
	 *
	 * @since 1.0.0
	 *
	 * @access private
	 * @static
	 *
	 * @var MagicAI_GoogleTTS The single instance of the class.
	 */
	private static $_instance = null;

    /**
	 * Instance
	 *
	 * Ensures only one instance of the class is loaded or can be loaded.
	 *
	 * @since 1.0.0
	 *
	 * @access public
	 * @static
	 *
	 * @return MagicAI_GoogleTTS An instance of the class.
	 */
	public static function instance() {

		if ( is_null( self::$_instance ) ) {
			self::$_instance = new self();
		}
		return self::$_instance;
    }

    /**
	 * Constructor
	 *
	 * @since 1.0.0
	 *
	 * @access public
	 */
    public function __construct() {
       
        $this->init();

    }

    /**
	 * Initialize
	 *
	 * Load the files required to run the plugin.
	 *
	 * @since 1.0.0
	 *
	 * @access public
	 */
    public function init() {

		$this->load_options();
		$this->hooks();
        $this->include_files();

    }

    /**
     * Load Options
     *
     * This function loads the MagicAI plugin options from the WordPress database.
     * It retrieves the settings.
     *
     * @since 1.0.0
     *
     * @access public
     */
    public function load_options() {

        // Retrieve the MagicAI settings from the database
        $this->gcs_file = ! empty( magicai_helper()->get_option( 'gcs_file' ) ) ? get_attached_file(attachment_url_to_postid( magicai_helper()->get_option( 'gcs_file' ) )) : '';
        $this->gcs_name = magicai_helper()->get_option( 'gcs_name' );

    }

    /**
     * Hooks Registration
     *
     * This function registers WordPress action and filter hooks for the MagicAI plugin.
     * It allows you to define custom actions and filters to extend or modify the plugin's functionality.
     *
     * @since 1.0.0
     *
     * @access public
     */
	public function hooks() {

        add_action( 'wp_ajax_magicai_googletts', [ $this, 'sendRequest' ] );

	}
    
    /**
     * Include Required Files
     *
     * This function includes the necessary PHP files and dependencies for the MagicAI plugin to function correctly.
     * It ensures that all required files are loaded for proper execution of the plugin's features.
     *
     * @since 1.0.0
     *
     * @access public
     */
    public function include_files() {
        
    }

    public function sendRequest() {

        MagicAI_License::instance()->ajax_message();

        $speeches = $_POST['speeches'];

        if ( empty( $speeches ) || $speeches == '[]' ) {
            wp_send_json( [
                'error' => true,
                'message' => esc_html__( 'Speeches are empty, Please fill and send a new request again!', 'magicai-wp' ),
            ] );
        }

        try {
            $client = new TextToSpeechClient([
                'credentials' => $this->gcs_file,
                'project_id' => $this->gcs_name,
            ]);
        } catch (\Exception $e) {
            // Connection error occurred
            MagicAI_Logs::instance()->add_log(
                'voiceover',
                'Error',
                $e->getMessage(),
            );
            wp_send_json( [
                'error' => true,
                'message' => $e->getMessage()
            ] );
        }
        
        $wordCount = 0;
        $post_content = '';
        $ssml = '<speak>';
        foreach (json_decode( stripslashes($speeches), true )  as $speech) {

            $ssml .= sprintf(
                '<lang xml:lang="%3$s">
                    <prosody rate="%4$s">
                        <voice name="%1$s">%2$s</voice>
                        <break time="%5$ss"/>
                    </prosody>
                </lang>',
                $speech['voice'],
                $speech['content'],
                $speech['lang'],
                $speech['pace'],
                $speech['break'],
            );

            $post_content .= sprintf(
                '<p>%1$s %2$s: %3$s</p>',
                magicai_helper()->country2flag($speech['lang']),
                magicai_helper()->get_const_vars('GOOGLE_TTS_VOICES')[$speech['voice']],
                $speech['content'],
            );

        }
        $ssml .= '</speak>';

        // Set the SSML as the synthesis input
        $synthesisInputSsml = (new SynthesisInput())
            ->setSsml($ssml);

        // Build the voice request, select the language code ("en-US") and the ssml voice gender
        $voice = (new VoiceSelectionParams())
            ->setLanguageCode('en-US')
            ->setSsmlGender(SsmlVoiceGender::FEMALE);

        // select the type of audio file you want returned
        $audioConfig = (new AudioConfig())
            ->setAudioEncoding(AudioEncoding::MP3);

        // Perform text-to-speech request on the SSML input with selected voice parameters and audio file type
        $response = $client->synthesizeSpeech($synthesisInputSsml, $voice, $audioConfig);
        $audioContent = $response->getAudioContent();

        $client->close();

        MagicAI_Logs::instance()->add_log(
            'voiceover',
            'Audio Generated',
            wp_json_encode( [
                'data' => $ssml,
            ] ),
            1
        );

        $post_id = wp_insert_post( [
            'post_type' => 'magicai-documents',
            'post_status' => 'publish',
            'post_content' => $post_content,
            'post_title' => 'output.mp3',
            'meta_input' => [ '_magicai_doc_type' => 'voiceover', '_magicai_attachment_id' => 6215, '_magicai_userid' => get_current_user_id() ],
        ] );

        if ( is_wp_error( $post_id ) ) {
            wp_send_json( [
                'error' => true,
                'message' =>  $post_id->get_error_message(),
            ] );
        }

        $output = sprintf( 
            '<div class="doc-item">
                <div class="doc-item--title">%3$s %1$s</div>
                <div class="doc-item--content">
                    %4$s
                    <div class="data-audio" data-audio="%2$s">
                        <div class="audio-preview"></div>
                    </div>
                </div>
            </div>',
            'output.mp3',
            6215,
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><path d="M12.4142 5H21C21.5523 5 22 5.44772 22 6V20C22 20.5523 21.5523 21 21 21H3C2.44772 21 2 20.5523 2 20V4C2 3.44772 2.44772 3 3 3H10.4142L12.4142 5ZM4 5V19H20V7H11.5858L9.58579 5H4ZM11 13.05V9H16V11H13V15.5C13 16.8807 11.8807 18 10.5 18C9.11929 18 8 16.8807 8 15.5C8 14.1193 9.11929 13 10.5 13C10.6712 13 10.8384 13.0172 11 13.05Z"></path></svg>',
            $post_content,
        );

        wp_send_json( [
            'output' => $output
        ] );
    }

}
MagicAI_GoogleTTS::instance();
