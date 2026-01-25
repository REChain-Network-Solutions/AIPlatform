<?php 

if ( ! defined( 'ABSPATH' ) ) exit; // Exit if accessed directly

use Aws\S3\S3Client;

#[AllowDynamicProperties]
class MagicAI_Amazon_S3 {

    /**
	 * Instance
	 *
	 * @since 1.0.0
	 *
	 * @access private
	 * @static
	 *
	 * @var MagicAI_Amazon_S3 The single instance of the class.
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
	 * @return MagicAI_Amazon_S3 An instance of the class.
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

        if ( magicai_helper()->get_option('storage', 'wp') == 's3') {
            // Instantiate an Amazon S3 client.
            $this->s3client = new S3Client([
                'version' => 'latest',
                'region'  => magicai_helper()->get_option('aws_default_region', 'eu-north-1'),
                'credentials' => [
                    'key' => magicai_helper()->get_option('aws_access_key_id', '123'),
                    'secret' => magicai_helper()->get_option('aws_secret_access_key', '123'),
                ],
                'use_path_style_endpoint' => false,
            ]);
        }

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

	}

    /**
     * Uploads an image to an AWS S3 bucket.
     *
     * @since 1.0.0
     *
     * @param string $file_name  The name of the file to be uploaded.
     * @param string $file       The content of the file to be uploaded.
     * @param string $file_type  Optional. The type of the file. Default is 'png'.
     *
     * @return string|bool       The URL of the uploaded image on success, false on failure.
     */
    function upload_image( $file_name, $file, $file_type = 'png' ) {
        try {
            $result = $this->s3client->putObject([
                'Body' => $file,
                'Bucket' => magicai_helper()->get_option('aws_bucket'),
                'Key' => $file_name,
                'ContentType' => "image/$file_type",
            ]);
            return $result['ObjectURL'];
        } catch (Exception $exception) {
            wp_send_json( [
                'error' => true,
                'message' => $exception->getMessage(),
            ] );
        }
    }

    /**
     * Deletes an image from an AWS S3 bucket.
     *
     * @since 1.0.0
     *
     * @param string $file_name  The name of the file to be deleted.
     */
    function delete_image( $file_name ) {
        $this->s3client->deleteObject([
            'Bucket' => magicai_helper()->get_option('aws_bucket'),
            'Key'    => $file_name
        ]);
    }

}
MagicAI_Amazon_S3::instance();
