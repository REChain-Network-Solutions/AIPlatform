<?php
/**
 * The Helper
 * Contains all the helping functions
 *
 */

class MagicAI_Helper {

	/**
	 * Hold an instance of MagicAI_Helper class.
	 * @var MagicAI_Helper
	 */
	protected static $instance = null;

	/**
	 * Main MagicAI_Helper instance.
	 *
	 * @return MagicAI_Helper - Main instance.
     * 
     * @since 1.0.0
	 */
	public static function instance() {

		if(null == self::$instance) {
			self::$instance = new MagicAI_Helper();
		}

		return self::$instance;
	}

    /**
     * Retrieve the value of a specific option from the 'magicai_options' array.
     *
     * @param string $option_name The name of the option to retrieve.
     * @return mixed|null The value of the specified option if it exists, or null if the option is not set.
     * 
     * @since 1.0.0
     */
    function get_option( $option_name, $default = null ) {

        $options = get_option( 'magicai_options', [] );
        if ( isset( $options[ $option_name ] ) ) {
            return $options[ $option_name ];
        } 

        if ( $default ) {
            return $default;
        }

    }

    /**
     * Generates HTML for a select dropdown of OpenAI languages.
     *
     * @return string HTML for the select dropdown of languages.
     * 
     * @since 1.0.0
     */
    function get_openai_languages_html() {
        
        $html = '<select name="language" id="language" required>';
            foreach ( magicai_helper()->get_const_vars('OPENAI_LANGUAGES') as $language_code => $language_name ) {
                $html .= sprintf( 
                    '<option value="%1$s" %2$s>%3$s</option>',
                    esc_attr( $language_code ),
                    selected( $language_code, esc_attr( magicai_helper()->get_option( 'openai_default_language' ) ), false ),
                    esc_html__( $language_name )
                );
            }
        $html .='</select>';

        return $html;
    }

    /**
     * Detect the programming language of a given code snippet.
     *
     * @param string $code The code snippet to be analyzed.
     *
     * @return string|false The detected programming language or `false` if no language is confidently detected.
     * 
     * @since 1.0.0
     */
    function detect_programming_language($code) {
        // Define an array of unique keywords or patterns for each programming language
        $languages = [
            'php' => ['<?php', 'echo', 'function', 'class', 'require_once'],
            'python' => ['print', 'def', 'class', 'import'],
            'javascript' => ['function', 'var', 'console.log', 'import'],
            'java' => ['public', 'class', 'void', 'import'],
            'c++' => ['#include', 'int', 'class', 'public'],
            'c#' => ['using', 'class', 'public', 'void'],
            'ruby' => ['puts', 'def', 'class', 'require'],
            'swift' => ['import', 'func', 'class', 'var'],
            'go' => ['package', 'func', 'import', 'var'],
            'rust' => ['fn', 'struct', 'use', 'impl'],
            'scala' => ['class', 'def', 'val', 'import'],
            'kotlin' => ['fun', 'class', 'import', 'val'],
            'perl' => ['print', 'sub', 'package', 'use'],
            'lua' => ['print', 'function', 'local', 'require'],
            'r' => ['print', 'function', 'library', 'data.frame'],
            'html' => ['<html', '<head', '<body', '<div'],
            'css' => ['body', 'div', 'margin', 'font-size'],
            'sql' => ['SELECT', 'FROM', 'WHERE', 'JOIN'],
            'typescript' => ['const', 'interface', 'import', 'class'],
            'dart' => ['void', 'class', 'import', 'main'],
        ];
    
        // Initialize an array to store the count of keywords for each language
        $languageScores = [];
    
        // Iterate through each language's keywords and count how many are found in the code
        foreach ($languages as $language => $keywords) {
            $score = 0;
            foreach ($keywords as $keyword) {
                if (stripos($code, $keyword) !== false) {
                    $score++;
                }
            }
            $languageScores[$language] = $score;
        }
    
        // Find the language with the highest score
        $detectedLanguage = array_search(max($languageScores), $languageScores);
    
        return $detectedLanguage;
    }
    

    /**
     * Insert an image into the WordPress media library and add custom metadata.
     *
     * @param string $image_url The URL of the image to be inserted.
     * @param string $prompt    The prompt or additional information for the image.
     *
     * @return void
     * 
     * @since 1.0.0
     */
    function insert_image_to_gallery($image_url, $prompt, $base_64 = false, $return_attachment_id = false) {
        // Get the path to the uploads directory
        $upload_dir = wp_upload_dir();
        
        // Get the image data from the provided URL
        if ( $base_64 ) {
            $image_data = $image_url;
            // Generate a sanitized filename for the image
            $filename = sanitize_file_name(uniqid( 'magicai-' )) . '.png';
            // Get the attachment ID for the image
            $file_type = 'image/png';
        } else {
            $image_data = file_get_contents($image_url);
            // Generate a sanitized filename for the image
            $filename = sanitize_file_name(uniqid( 'magicai-' )) . '.png';
            $file_type = 'image/png';
        }
        
        // Save the image to the uploads directory
        if (wp_mkdir_p($upload_dir['path'])) {
            $file = $upload_dir['path'] . '/' . $filename;
        } else {
            $file = $upload_dir['basedir'] . '/' . $filename;
        }
        
        // Write the image data to the file
        file_put_contents($file, $image_data);
        
        $attachment = array(
            'post_mime_type' => $file_type,
            'post_title'     => sanitize_file_name(str_replace('.png', '', $filename)),
            'post_content'   => '',
            'post_status'    => 'inherit'
        );
        
        $attachment_id = wp_insert_attachment($attachment, $file, $post_id); // You need to provide $post_id.
        
        // Include necessary files
        require_once(ABSPATH . 'wp-admin/includes/image.php');
        
        // Generate attachment metadata
        $attachment_data = wp_generate_attachment_metadata($attachment_id, $file);
        
        // Update attachment metadata
        wp_update_attachment_metadata($attachment_id, $attachment_data);
        
        // Update attachment metadata with custom values
        update_post_meta($attachment_id, '_magicai_userid', get_current_user_id());
        update_post_meta($attachment_id, '_magicai_prompt', $prompt);

        if ( $return_attachment_id ) {
            return $attachment_id;
        }
    }

    /**
     * Insert an image into the Amazon S3 and add custom metadata.
     *
     * @param string $image_url The URL of the image to be inserted.
     * @param string $prompt    The prompt or additional information for the image.
     *
     * @return void
     * 
     * @since 1.0.0
     */
    function insert_image_to_s3($image_url, $prompt, $base_64 = false, $basename = false) {
        
        // Get the image data from the provided URL
        if ( $base_64 ) {
            $image_data = base64_decode($image_url);
        } else {
            $image_data = file_get_contents($image_url);
        }
        $filename = sanitize_file_name(uniqid( 'magicai-' )) . '.png';
        $file_type = 'image/png';

        $s3_url = MagicAI_Amazon_S3::instance()->upload_image( $filename, $image_data );

        $post_id = wp_insert_post( [
            'post_type' => 'magicai-attachments',
            'post_title' => $filename,
            'post_content' => $s3_url,
            'post_author' => get_current_user_id(),
            'post_status' => 'publish'
        ] );

        if ( is_wp_error( $post_id ) ) {
            wp_send_json( [
                'error' => true,
                'message' => $post_id->get_error_message(),
            ] );
        }

        // Update attachment metadata with custom values
        update_post_meta($post_id, '_magicai_userid', get_current_user_id());
        update_post_meta($post_id, '_magicai_prompt', $prompt);

        return $post_id;
    }

    /**
     * Insert an image into the WordPress media library and add custom metadata.
     *
     * @param string $image_url The URL of the image to be inserted.
     * @param string $post_id   The post_id for to the insert thumbnail.
     *
     * @return void
     * 
     * @since 1.0.0
     */
    function insert_image_to_post( $post_id, $image_url ) {

        // Get the path to the uploads directory
        $upload_dir = wp_upload_dir();
        $image_data = file_get_contents($image_url);
        // $filename = basename($image_url);
        $filename = sanitize_file_name(parse_url($image_url)['path']);

        if (strpos($filename, '.jpg') === false) {
            $filename = "$filename.jpg";
        }
    
        // Save the image to the uploads directory
        if ( wp_mkdir_p($upload_dir['path']) ) {
            $file = $upload_dir['path'] . '/' . $filename;
        } else {
            $file = $upload_dir['basedir'] . '/' . $filename;
        }
    
        file_put_contents($file, $image_data);
    
        // Get the attachment ID for the image
        $wp_filetype = wp_check_filetype($filename, null );
        $attachment = array(
            'post_mime_type' => $wp_filetype['type'],
            'post_title' => sanitize_file_name(str_replace('.jpg','', $filename)),
            'post_content' => '',
            'post_status' => 'inherit'
        );
        $attachment_id = wp_insert_attachment( $attachment, $file, $post_id );
        require_once(ABSPATH . 'wp-admin/includes/image.php');
        $attachment_data = wp_generate_attachment_metadata( $attachment_id, $file );
        wp_update_attachment_metadata( $attachment_id, $attachment_data );
    
        // Set the attachment ID as the featured image for the post
        set_post_thumbnail($post_id, $attachment_id);
    
    }

    /**
     * Get images associated with attachments that have a specific '_magicai_prompt' meta key.
     *
     * This function retrieves image URLs for attachments that have the specified '_magicai_prompt' meta key.
     *
     * @return array An array of image URLs for matching attachments.
     * 
     * @since 1.0.0
     */
    function get_attachment_images( $atts = null ) {

        $meta_query = array();

        if ( isset( $atts['user_filter'] ) ) {
            $filter = $atts['user_filter'];

            $meta_query = array(
                'key' => '_magicai_userid',
                'value' => '',
                'compare' => '=',
                'type' => 'NUMERIC',
            );

            if ( $filter == 'false' || empty( $filter ) ) {
                $meta_query = array();
            } elseif ( $filter == 'true' ) {
                $meta_query['value'] = get_current_user_id();
            } else {
                $meta_query['value'] = $filter;
            }
        }

        $images = '';
        $attachments = get_posts( array(
            'post_type' => 'attachment',
            'posts_per_page' => -1,
            'meta_query' => array(
                'relation' => 'AND',
                array(
                    'key' => '_magicai_prompt',
                    'compare' => 'EXISTS',
                ),
                $meta_query,
            ),
        ) );
 
        $magicai_attachments = get_posts( array(
            'post_type' => 'magicai-attachments',
            'posts_per_page' => -1,
            'meta_query' => array(
                'relation' => 'AND',
                array(
                    'key' => '_magicai_prompt',
                    'compare' => 'EXISTS',
                ),
                $meta_query,
            ),
        ) );

        $attachments = array_merge( $attachments, $magicai_attachments ); // Merge
        usort($attachments, function($a, $b) { // Sort by date
            return $b->ID - $a->ID;
        });

        if ( $attachments ) {
            foreach( $attachments as $attachment ) {

                if ( $attachment->post_type == 'magicai-attachments' ) {
                    $images .= sprintf( 
                        '<figure class="gallery-item" data-src="%2$s" data-prompt="%3$s" data-name="%4$s" data-id="%5$s">%1$s</figure>',
                        '<img src="'.$attachment->post_content.'" loading="lazy" class="gallery-img">',
                        esc_url( $attachment->post_content ),
                        get_post_meta( $attachment->ID, '_magicai_prompt', true ),
                        basename( $attachment->post_content ),
                        esc_attr($attachment->ID)
                    );    
                } else {
                    $images .= sprintf( 
                        '<figure class="gallery-item" data-src="%2$s" data-prompt="%3$s" data-name="%4$s" data-id="%5$s">%1$s</figure>',
                        wp_get_attachment_image( $attachment->ID, 'full', '', [ 'class' => 'gallery-img', 'loading' => 'lazy' ] ),
                        esc_url( wp_get_attachment_image_url( $attachment->ID, 'full' ) ),
                        get_post_meta( $attachment->ID, '_magicai_prompt', true ),
                        basename( get_attached_file( $attachment->ID ) ),
                        esc_attr( $attachment->ID )
                    );
                }
            }
        }

        echo $images;

    }

    /**
     * Retrieves a CURLFile object representing an image attachment.
     *
     * This function retrieves an image attachment from WordPress by its attachment ID,
     * creates a CURLFile object, and returns it. The CURLFile object can be used in
     * HTTP requests, such as file uploads via cURL.
     *
     * @param int $attachment_id The ID of the image attachment to retrieve.
     * @return CURLFile|false A CURLFile object representing the image attachment, or false
     * if the attachment could not be found or if there was an error.
     * 
     * @since 1.0.0
     */
    function get_image_curlfile( $attachment_id ) {

        $attachment_url = wp_get_attachment_url( $attachment_id );
        $attachment_type = get_post_mime_type( $attachment_id );
        $attachment_filename = basename( get_attached_file( $attachment_id ) );
        $cfile = new CURLFile( $attachment_url, $attachment_type, $attachment_filename );

        return $cfile;
    }

    /**
     * Retrieves a list of chat posts and organizes them by date.
     *
     * This function queries for chat posts of the 'magicai-chat' post type and displays
     * them grouped by date, with a date label for each date group. It also includes a
     * specific class for today's chats and a generic class for last week's chats.
     *
     * @return void
     * 
     * @since 1.0.0
     */
    function get_chat_list( $atts = null, $type = 'chat' ) {

        $meta_query = [
            'relation' => 'OR',
            [
                'key' => '_magicai_chat_type',
                'compare' => 'NOT EXISTS',
            ],
            [
                'key' => '_magicai_chat_type',
                'value' => 'chatbot',
                'compare' => '!=',
            ],
        ];

        if ( isset( $atts['user_filter'] ) ) {
            $filter = $atts['user_filter'];
            if ( ( $filter == 'true' || !empty( $filter ) ) && $filter != 'false' ) {
                if ( $type == 'vision' ){
                    $meta_query = [ 
                        'key' => '_magicai_userid',
                        'value' => $filter == 'true' ? get_current_user_id() : $filter,
                        'compare' => '=',
                        'type' => 'NUMERIC',
                    ];
                } else {
                    $meta_query = [ 
                        'relation' => 'AND',
                        $meta_query,
                        [
                            'key' => '_magicai_userid',
                            'value' => $filter == 'true' ? get_current_user_id() : $filter,
                            'compare' => '=',
                            'type' => 'NUMERIC',
                        ]
                    ];
                }
            }
        }

        $posts = get_posts( [
            'post_type' => "magicai-$type",
            'posts_per_page' => -1,
            'meta_query' => $meta_query
        ] );

        if ( $posts ) {
            $today = date('Y-m-d');
            $lastWeek = date('Y-m-d', strtotime('-7 days'));
            $currentDateLabel = '';  // Initialize the current date label
            $class = '';
        
            foreach ($posts as $post) {
                $postDate = date('Y-m-d', strtotime($post->post_date));
                $dateLabel = '';
        
                if ($postDate == $today) {
                    $dateLabel = esc_html__('Today','magicai-wp');
                    $class = 'today'; 
                } elseif ($postDate >= $lastWeek) {
                    $dateLabel = esc_html__('Last Week','magicai-wp');
                    $class = '';
                } else {
                    // You can add more date ranges or a default label here if needed
                    $dateLabel = esc_html__('Older','magicai-wp');
                }
        
                if ($dateLabel != $currentDateLabel) {
                    // Display the date label when it changes
                    printf( '<div class="magicai-chat--list-date %s">%s</div>', esc_attr($class), $dateLabel );
                    $currentDateLabel = $dateLabel;
                }
        
                ?>
                <div class="magicai-chat--list-chat">
                    <div class="magicai-chat--list-chat--trigger" data-postid="<?php esc_attr_e($post->ID); ?>"></div>
                    <div class="icon">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18">
                            <path d="M5.76282 17H20V5H4V18.3851L5.76282 17ZM6.45455 19L2 22.5V4C2 3.44772 2.44772 3 3 3H21C21.5523 3 22 3.44772 22 4V18C22 18.5523 21.5523 19 21 19H6.45455Z"></path>
                        </svg>
                    </div>
                    <div class="message"><?php esc_html_e($post->post_title); ?></div>
                    <div class="dropdown">
                        <details>
                            <summary>
                            <div class="dropdown-trigger">
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" width="18" height="18">
                                    <path d="M10 3a1.5 1.5 0 110 3 1.5 1.5 0 010-3zM10 8.5a1.5 1.5 0 110 3 1.5 1.5 0 010-3zM11.5 15.5a1.5 1.5 0 10-3 0 1.5 1.5 0 003 0z" />
                                </svg>
                            </div>
                            </summary>
                            <div class="dropdown-content">
                                <span class="magicai-chat--list-chat--action" data-action="edit" data-postid="<?php esc_attr_e($post->ID); ?>">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M6.41421 15.89L16.5563 5.74786L15.1421 4.33365L5 14.4758V15.89H6.41421ZM7.24264 17.89H3V13.6474L14.435 2.21233C14.8256 1.8218 15.4587 1.8218 15.8492 2.21233L18.6777 5.04075C19.0682 5.43128 19.0682 6.06444 18.6777 6.45497L7.24264 17.89ZM3 19.89H21V21.89H3V19.89Z"></path></svg>
                                    <?php esc_html_e( 'Edit', 'magicai-wp' ); ?>
                                </span>
                                <span class="magicai-chat--list-chat--action" data-action="delete" data-postid="<?php esc_attr_e($post->ID); ?>">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M4 8H20V21C20 21.5523 19.5523 22 19 22H5C4.44772 22 4 21.5523 4 21V8ZM6 10V20H18V10H6ZM9 12H11V18H9V12ZM13 12H15V18H13V12ZM7 5V3C7 2.44772 7.44772 2 8 2H16C16.5523 2 17 2.44772 17 3V5H22V7H2V5H7ZM9 4V5H15V4H9Z"></path></svg>
                                    <?php esc_html_e( 'Delete', 'magicai-wp' ); ?>
                                </span>
                            </div>
                        </details>
                    </div>
                </div>
                <?php
            }
        }
    }

    /**
     * Get documents data of a specific type.
     *
     * This function retrieves documents of the specified type and generates HTML for display.
     *
     * @param string $type The type of documents to retrieve.
     * @return string HTML representation of the documents data.
     *
     * @since 1.0.0
     */
    function get_documents_data( $type = null ) {

        if ( ! $type ) { return; }

        $posts = get_posts(  
            [
                'post_type' => 'magicai-documents',
                'posts_per_page' => -1,
                'meta_key' => '_magicai_doc_type',
                'meta_value' => $type,
            ]
        );

        $html = '';
        if ( $type == 'transcribe' ) {
            foreach ( $posts as $post ) {
                $html .= sprintf( 
                    '<div class="doc-item">
                        <div class="doc-item--title">%3$s %1$s</div>
                        <div class="doc-item--content">%2$s</div>
                    </div>',
                    $post->post_title,
                    $post->post_content,
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><path d="M12.4142 5H21C21.5523 5 22 5.44772 22 6V20C22 20.5523 21.5523 21 21 21H3C2.44772 21 2 20.5523 2 20V4C2 3.44772 2.44772 3 3 3H10.4142L12.4142 5ZM4 5V19H20V7H11.5858L9.58579 5H4ZM11 13.05V9H16V11H13V15.5C13 16.8807 11.8807 18 10.5 18C9.11929 18 8 16.8807 8 15.5C8 14.1193 9.11929 13 10.5 13C10.6712 13 10.8384 13.0172 11 13.05Z"></path></svg>'
                );
            }
        } elseif ( $type == 'voiceover' ) {
            foreach ( $posts as $post ) {
                $html .= sprintf( 
                    '<div class="doc-item">
                        <div class="doc-item--title">%3$s %1$s</div>
                        <div class="doc-item--content">
                            %4$s
                            <div class="data-audio" data-audio="%2$s">
                                <div class="audio-preview"></div>
                            </div>
                        </div>
                    </div>',
                    $post->post_title,
                    wp_get_attachment_url( get_post_meta( $post->ID, '_magicai_attachment_id', true ) ),
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><path d="M12.4142 5H21C21.5523 5 22 5.44772 22 6V20C22 20.5523 21.5523 21 21 21H3C2.44772 21 2 20.5523 2 20V4C2 3.44772 2.44772 3 3 3H10.4142L12.4142 5ZM4 5V19H20V7H11.5858L9.58579 5H4ZM11 13.05V9H16V11H13V15.5C13 16.8807 11.8807 18 10.5 18C9.11929 18 8 16.8807 8 15.5C8 14.1193 9.11929 13 10.5 13C10.6712 13 10.8384 13.0172 11 13.05Z"></path></svg>',
                    $post->post_content,
                );
            }
        }
    
        return $html;
    
    }


    /**
     * Retrieves HTML for displaying chatbot templates.
     *
     * Retrieves posts of type 'magicai-chatbot' and generates HTML for each post,
     * including title, actions for editing and deleting, and associated meta.
     *
     * @return string HTML content for chatbot templates.
     * 
     * @since 1.0.0
     */
    function get_chatbot_templates() {

        $posts = get_posts(  
            [
                'post_type' => 'magicai-chatbot',
                'posts_per_page' => -1,
                'post_status' => 'any',
            ]
        );

        $html = '';
        
        if ( ! $posts ){ return; }

        $html .= sprintf(
            '<table class="widefat striped">
                <thead>
                    <th>%1$s</th>
                    <th>%2$s</th>
                    <th>%3$s</th>
                    <th>%4$s</th>
                    <th>%5$s</th>
                </thead><tbody>',
            esc_html__( 'Image', 'magicai-wp' ),
            esc_html__( 'Name', 'magicai-wp' ),
            esc_html__( 'Role', 'magicai-wp' ),
            esc_html__( 'Conversations', 'magicai-wp' ),
            esc_html__( 'Actions', 'magicai-wp' ),
        );

        foreach ( $posts as $post ) {
            $html .= sprintf( 
                '<tr>
                    <td><img src="%6$s" width="28" height="28"/></td>
                    <td>%1$s</td>
                    <td>%7$s</td>
                    <td>%9$s</td>
                    <td>
                        <div class="actions">
                            <a href="%2$s" title="%4$s"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M15.7279 9.57629L14.3137 8.16207L5 17.4758V18.89H6.41421L15.7279 9.57629ZM17.1421 8.16207L18.5563 6.74786L17.1421 5.33365L15.7279 6.74786L17.1421 8.16207ZM7.24264 20.89H3V16.6474L16.435 3.21233C16.8256 2.8218 17.4587 2.8218 17.8492 3.21233L20.6777 6.04075C21.0682 6.43128 21.0682 7.06444 20.6777 7.45497L7.24264 20.89Z"></path></svg></a>
                            <a href="%3$s" title="%5$s" onclick="return confirm(\'%8$s\')"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M17 6H22V8H20V21C20 21.5523 19.5523 22 19 22H5C4.44772 22 4 21.5523 4 21V8H2V6H7V3C7 2.44772 7.44772 2 8 2H16C16.5523 2 17 2.44772 17 3V6ZM18 8H6V20H18V8ZM9 11H11V17H9V11ZM13 11H15V17H13V11ZM9 4V6H15V4H9Z"></path></svg></a>
                        </div>
                    </td>
                </tr>',
                esc_html( $post->post_title ),
                get_edit_post_link( $post->ID ),
                get_delete_post_link( $post->ID ),
                esc_html__( 'Edit', 'magicai-wp' ),
                esc_html__( 'Delete', 'magicai-wp' ),
                get_post_meta( $post->ID, '_image', true ),
                get_post_meta( $post->ID, '_role', true ),
                esc_js( __('Are you sure?', 'magicai-wp') ),
                magicai_helper()->get_chatbot_conversation_count( $post->ID ),
            );
        }

        $html .= '</tbody></table>';

        return $html;

    }

    /**
     * Retrieves the count of conversations associated with a specific chatbot template.
     *
     * This function fetches the count of conversations from the 'magicai-chat' post type
     * that are linked to a specified chatbot template ID.
     *
     * @since 1.3
     *
     * @param int|null $template_id Optional. The ID of the chatbot template. Default is null.
     * @return int The count of conversations associated with the specified chatbot template.
     */
    function get_chatbot_conversation_count( $template_id = null ) {

        $chats = get_posts( [
            'post_type' => "magicai-chat",
            'posts_per_page' => -1,
            'meta_query' => [
                [
                    'key' => '_magicai_chatbot_template_id',
                    'value' => $template_id,
                    'compare' => '=',
                ]
            ]
        ] );

        return count( $chats );

    }

    /**
     * Define custom MIME types.
     *
     * This function defines custom MIME types as an associative array where keys are file extensions, and values are MIME types.
     *
     * @return array An associative array of custom MIME types.
     *
     * @since 1.0.0
     */
    function custom_mime_types() {
        return [
            'mp3'  => 'audio/mpeg',
            'mp4'  => 'video/mp4',
            'mpeg' => 'video/mpeg',
            'mpga' => 'audio/mpeg',
            'm4a'  => 'audio/mp4',
            'wav'  => 'audio/wav',
            'webm' => 'video/webm',
            'json' => 'application/json',
        ];
    }

    /**
     * Convert a country code to its corresponding flag emoji.
     *
     * @param string $countryCode The country code to convert.
     * @return string The flag emoji corresponding to the given country code.
     * 
     * @since 1.0.0
     */
    function country2flag(string $countryCode) {

        if (strpos($countryCode, '-') !== false) {
            $countryCode = substr($countryCode, strpos($countryCode, '-') + 1);
        } elseif (strpos($countryCode, '_') !== false) {
            $countryCode = substr($countryCode, strpos($countryCode, '_') + 1);
        }

        if ( $countryCode === 'el' ){
            $countryCode = 'gr';
        }elseif ( $countryCode === 'da' ){
            $countryCode = 'dk';
        }
        
        return (string) preg_replace_callback(
            '/./',
            static fn (array $letter) => mb_chr(ord($letter[0]) % 32 + 0x1F1E5),
            $countryCode
        );
    }

    /**
     * Get the corresponding status classname based on a log message.
     *
     * @param string $status The status to check.
     * @return string The classname corresponding to the given status message.
     * 
     * @since 1.0.0
     */
    function get_log_status_classname_by_message($message) {
        $messages = [
            'success' => ['Post Created', 'Post Duplicated', 'Post Saved as Draft', 'Audio Generated', 'Completion Worked', 'Code Generated', 'Image Generated', 'Transcribe Generated'],
            'error'   => ['Post not Created', 'Error', 'Completion Failed', 'Post Created', 'Failed'],
            'default' => ['Chat Deleted', 'Chat Name Updated', 'Post Updated'],
        ];

        foreach ($messages as $key => $values) {
            if (in_array($message, $values)) {
                return $key;
            }
        }

        // Return a default value if no match is found
        return 'default';

    }

    /**
     * Display a help tip with the provided text.
     *
     * This function generates HTML markup for a help tip, which includes an information
     * icon and the specified text.
     *
     * @param string $text The text to be displayed in the help tip.
     * @return void
     * 
     * @since 1.0.0
     */
    function label_help_tip( $text, $print = true ) {
        if ( $text ) {
           $content = sprintf( 
                '<div class="magicai-help-tip">
                    <div class="trigger">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12C22 17.5228 17.5228 22 12 22ZM11 11V17H13V11H11ZM11 7V9H13V7H11Z"></path></svg>
                    </div>
                    <div class="text">
                        %s
                    </div>
                </div>',
                esc_html__( $text, 'magicai-wp' ) 
            );

            if ( $print ) {
                echo $content;
            } else {
                return $content;
            }
        }
    }

    /**
     * Generators default HTML template.
     *
     * @since 1.0.0
     */
    function generator_default_template( $text = null ) {
        ?>
            <div class="result-default">
                <img src="<?php echo esc_url(MAGICAI_URL . 'assets/img/logo.svg');?>" width="120"> 
                <p>
                    <?php 
                        if ( empty( $text ) ) {
                            esc_html_e( 'Fill the form and click the Generate button', 'magicai-wp' );
                        } else {
                            echo $text;
                        }
                    ?>
                </p>
            </div>
        <?php
    }

    /**
     * Removes quotes from the beginning and end of a string if they exist.
     *
     * @param string $title The input string that may contain quotes.
     *
     * @return string The modified string with quotes removed if they existed.
     * 
     * @since 1.0.0
     */
    function remove_quotes( $title ) {
        // Check if the title starts and ends with quotes
        if ( substr($title, 0, 1) === '"' && substr($title, -1) === '"' ) {
            // If yes, remove the quotes
            $title = substr( $title, 1, -1 );
        }
        
        return $title;
    }

    /**
     * Provides default prompts for an image generator.
     *
     * @return string A randomly selected prompt for the image generator.
     * 
     * @since 1.0.0
     */
    function image_generator_prompt_example() {
        $default = [
            esc_html__( 'A fire-breathing dragon wearing a top hat and sunglasses', 'magicai-wp' ),
            esc_html__( 'A serene beach at sunset with palm trees swaying in the breeze', 'magicai-wp' ),
            esc_html__( 'A futuristic cityscape with flying cars and holographic billboards', 'magicai-wp' ),
            esc_html__( 'A cozy library with floor-to-ceiling bookshelves and a fireplace', 'magicai-wp' ),
            esc_html__( 'A playful kitten riding a skateboard down a winding road', 'magicai-wp' ),
            esc_html__( 'An alien landscape with towering mushrooms and bioluminescent plants', 'magicai-wp' ),
            esc_html__( 'A steampunk-style airship sailing through the clouds', 'magicai-wp' ),
            esc_html__( 'A whimsical underwater world with talking fish and mermaids', 'magicai-wp' ),
            esc_html__( 'A grand castle atop a mountain with a cascading waterfall in the background', 'magicai-wp' ),
            esc_html__( 'A cyberpunk hacker in a neon-lit room surrounded by floating computer code', 'magicai-wp' ),
        ];

        return $default[array_rand($default, 1)];

    }

    /**
	 * Gets current user IP address.
	 *
	 * @param  bool $check_all_headers Check all headers? Default is `true`.
	 *
	 * @return string Current user IP address.
     * 
     * @since 1.0.0
	 */
	function get_ip( $check_all_headers = true ) {
		foreach (array('HTTP_CLIENT_IP', 'HTTP_X_FORWARDED_FOR', 'HTTP_X_FORWARDED', 'HTTP_X_CLUSTER_CLIENT_IP', 'HTTP_FORWARDED_FOR', 'HTTP_FORWARDED', 'REMOTE_ADDR') as $key){
			if (array_key_exists($key, $_SERVER) === true){
				foreach (explode(',', $_SERVER[$key]) as $ip){
					$ip = trim($ip);
	 
					if (filter_var($ip, FILTER_VALIDATE_IP, FILTER_FLAG_NO_PRIV_RANGE | FILTER_FLAG_NO_RES_RANGE) !== false){
						return $ip;
					}
				}
			}
		}
	}

    /**
     * Parse an RSS feed and retrieve posts.
     *
     * @param string $feed_url The URL of the RSS feed to parse.
     * @param int $limit (Optional) The maximum number of posts to retrieve. Default is 10.
     *
     * @return array|false Returns an array of posts with keys 'id', 'link', 'title', 'description', and 'image' if successful. Otherwise, returns false.
     */
    function parseRSS( $feed_url, $limit = 10 ) {
        $rss = simplexml_load_file($feed_url);
        $posts = array();

        if ($rss) {
            foreach ($rss->channel->item as $item) {
                $id = wp_unique_id();
                if ( $id > $limit ) break;
                $posts[] = [
                    'id' => $id,
                    'link' => $item->link,
                    'title' => $item->title,
                    'description' => $item->description,
                    'image' => $item->enclosure ? $item->enclosure['url'] : null
                ];
            }
            return $posts;
        }
        return false;
    }

    /**
     * Retrieves a unique salt value for use in cryptographic operations.
     *
     * This function fetches a salt value from WordPress Transients. If no salt value
     * is found, a new unique identifier (UID) is generated and stored as a transient
     * for subsequent use. The transient expiration time is set to 10 minutes.
     *
     * @return string The retrieved or newly generated salt value.
     * 
     * @since 1.0.0
     */
    function magicai_salt() {
        $salt = get_transient( 'magicai_salt' );
        if ( $salt === false ) {
            $salt = uniqid('',true);
            set_transient( 'magicai_salt', $salt, 10 * MINUTE_IN_SECONDS );
        }
        return $salt;
    }

    /**
     * Adds an image tag before each heading in the provided HTML.
     *
     * This function utilizes a regular expression to locate headings (h1-h6) in the HTML content
     * and adds an image tag before each found heading. The images are provided in an array and are
     * inserted sequentially before each heading.
     *
     * @param string $html    The HTML content where the image tags will be added before headings.
     * @param array  $images  An array containing URLs of images to be inserted before headings.
     * @return string The modified HTML content with image tags added before headings.
     * 
     * @since 1.0.0
     */
    function add_img_before_headings( $html, $images ) {
        shuffle($images);
        $html = preg_replace_callback('/(<h[1-6]>)/i', function($matches) use ($images) {
            static $index = 0;
            $currentImg = $images[$index];
            $index++;
            if ( magicai_helper()->get_option('storage', 'wp') == 's3' ) {
                $filename = sanitize_file_name(parse_url($currentImg)['path']);
                $filename = "$filename.jpg";
                $currentImg = MagicAI_Amazon_S3::instance()->upload_image( $filename, file_get_contents($currentImg), 'jpeg');
            }
            return '<img src="' . $currentImg . '">' . $matches[0];
        }, $html);

        return $html;
    }

    /**
     * Save data to the magicai-documents custom post type.
     *
     * This function creates a new document post with the provided data and saves it to the magicai-documents post type.
     *
     * @since 1.2
     *
     * @param array $data {
     *     An array of data for the document.
     *
     *     @type string $content     The content of the document.
     *     @type string $title       The title of the document.
     *     @type array  $tags        An array of tags for the document.
     *     @type string $type        The type of the document.
     * }
     */
    function save_to_documents( $data = array() ){

        if ( ! $data ) {
            return;
        }

        $document_id = wp_insert_post( [
            'post_type' => 'magicai-documents',
            'post_status' => 'publish',
            'post_content' => $data['content'],
            'post_title' => $data['title'],
            'post_tags' => $data['tags'],
            'meta_input' => [ 
                '_magicai_doc_type' => $data['type'],
                '_magicai_userid' => get_current_user_id()
            ],
        ] );

        if ( $data['tags'] ) {
            wp_set_post_terms( $document_id, $data['tags'], 'post_tag', false );
        }

        if ( is_wp_error( $document_id ) ) {
            wp_send_json( [
                'error' => true,
                'message' =>  $document_id->get_error_message(),
            ] );
        }

    }


    /**
     * Retrieve SERP (Search Engine Results Page) data using the Serper API.
     *
     * @since 1.2
     *
     * @param string $q The search query.
     *
     * @return void|array Returns void if the query parameter is empty.
     *                   Returns an array with an error message if the SerperAPI key is not set.
     *                   Returns the SERP data if the request is successful.
     */
    function get_serp_data( $q = '' ) {

        if ( !$q ) {
            return;
        }

        $api_key = magicai_helper()->get_option('serper_api_key');

        if ( ! $api_key ) {
            wp_send_json( [
                'error' => true,
                'message' => __('Fill the SerperAPI key first, please! Settings > Additional APIs > Serper.')
            ] );
        }

        $endpoint = 'https://google.serper.dev/search';
        $headers = array(
            'X-API-KEY' => $api_key,
            'Content-Type' => 'application/json',
        );

        $args = array(
            'body' => json_encode([ 'q' => $q ]),
            'headers' => $headers,
        );

        $response = wp_remote_post($endpoint, $args);

        if (is_wp_error($response)) {
            MagicAI_Logs::instance()->add_log(
                'error',
                'SerperAPI ERROR',
                $response->get_error_message(),
            );
            wp_send_json( [
                'error' => true,
                'message' => $response->get_error_message(),
            ] );
        } else {
            $body = wp_remote_retrieve_body($response);
            $json = json_decode( wp_remote_retrieve_body( $response ), true );
            if ( isset($json['statusCode']) ) {
                wp_send_json( [
                    'error' => true,
                    'message' => 'SERPER API: ' . $json['message'],
                ] );
            }
            return $body;
        }
    }

    function get_const_vars( $name ) {

        $const = [
            'OPENAI_LANGUAGES' => [
                'ar-AE' => esc_html__('Arabic', 'magicai-wp'),
                'cmn-CN'=> esc_html__('Chinese (Mandarin)', 'magicai-wp'),
                'cs-CZ' => esc_html__('Czech (Czech Republic)', 'magicai-wp'),
                'da-DK' => esc_html__('Danish (Denmark)', 'magicai-wp'),
                'de-DE' => esc_html__('German (Germany)', 'magicai-wp'),
                'el-GR' => esc_html__('Greek (Greece)', 'magicai-wp'),
                'en-US' => esc_html__('English (USA)', 'magicai-wp'),
                'es-ES' => esc_html__('Spanish (Spain)', 'magicai-wp'),
                'et-EE' => esc_html__('Estonian (Estonia)', 'magicai-wp'),
                'fi-FI' => esc_html__('Finnish (Finland)', 'magicai-wp'),
                'fr-FR' => esc_html__('French (France)', 'magicai-wp'),
                'he-IL' => esc_html__('Hebrew (Israel)', 'magicai-wp'),
                'hi-IN' => esc_html__('Hindi (India)', 'magicai-wp'),
                'hr-HR' => esc_html__('Croatian (Croatia)', 'magicai-wp'),
                'hu-HU' => esc_html__('Hungarian (Hungary)', 'magicai-wp'),
                'id-ID' => esc_html__('Indonesian (Indonesia)', 'magicai-wp'),
                'is-IS' => esc_html__('Icelandic (Iceland)', 'magicai-wp'),
                'it-IT' => esc_html__('Italian (Italy)', 'magicai-wp'),
                'ja-JP' => esc_html__('Japanese (Japan)', 'magicai-wp'),
                'kk-KZ' => esc_html__('Kazakh (Kazakhistan)', 'magicai-wp'),
                'ko-KR' => esc_html__('Korean (South Korea)', 'magicai-wp'),
                'lt-LT' => esc_html__('Lithuanian (Lithuania)', 'magicai-wp'),
                'ms-MY' => esc_html__('Malay (Malaysia)', 'magicai-wp'),
                'nb-NO' => esc_html__('Norwegian (Norway)', 'magicai-wp'),
                'nl-NL' => esc_html__('Dutch (Netherlands)', 'magicai-wp'),
                'pl-PL' => esc_html__('Polish (Poland)', 'magicai-wp'),
                'pt-BR' => esc_html__('Portuguese (Brazil)', 'magicai-wp'),
                'pt-PT' => esc_html__('Portuguese (Portugal)', 'magicai-wp'),
                'ro-RO' => esc_html__('Romanian (Romania)', 'magicai-wp'),
                'ru-RU' => esc_html__('Russian (Russia)', 'magicai-wp'),
                'sl-SI' => esc_html__('Slovenian (Slovenia)', 'magicai-wp'),
                'sv-SE' => esc_html__('Swedish (Sweden)', 'magicai-wp'),
                'sw-KE' => esc_html__('Swahili (Kenya)', 'magicai-wp'),
                'tr-TR' => esc_html__('Turkish (Turkey)', 'magicai-wp'),
                'vi-VN' => esc_html__('Vietnamese (Vietnam)', 'magicai-wp'),
            ],

            'OPENAI_TONE' => [
                esc_html__('Professional', 'magicai-wp'),
                esc_html__('Funny', 'magicai-wp'),
                esc_html__('Casual', 'magicai-wp'),
                esc_html__('Excited', 'magicai-wp'),
                esc_html__('Witty', 'magicai-wp'),
                esc_html__('Sarcastic', 'magicai-wp'),
                esc_html__('Feminine', 'magicai-wp'),
                esc_html__('Masculine', 'magicai-wp'),
                esc_html__('Bold', 'magicai-wp'),
                esc_html__('Dramatic', 'magicai-wp'),
                esc_html__('Grumpy', 'magicai-wp'),
                esc_html__('Secretive', 'magicai-wp'),
            ],

            'OPENAI_TEMPERATURE' => [
                '0.25' => esc_html__('Economic', 'magicai-wp'),
                '0.5' => esc_html__('Average', 'magicai-wp'),
                '0.75' => esc_html__('Good', 'magicai-wp'),
                '1' => esc_html__('Premium', 'magicai-wp'),
            ],

            'OPENAI_TTS_LANGUAGES' => [
                "af-ZA" => esc_html__( 'Afrikaans', 'magicai-wp'),
                "ar-SA" => esc_html__( 'Arabic', 'magicai-wp'),
                "hy-AM" => esc_html__( 'Armenian', 'magicai-wp'),
                "az-AZ" => esc_html__( 'Azerbaijani', 'magicai-wp'),
                "be-BY" => esc_html__( 'Belarusian', 'magicai-wp'),
                "bs-BA" => esc_html__( 'Bosnian', 'magicai-wp'),
                "bg-BG" => esc_html__( 'Bulgarian', 'magicai-wp'),
                "ca-ES" => esc_html__( 'Catalan', 'magicai-wp'),
                "zh-CN" => esc_html__( 'Chinese', 'magicai-wp'),
                "hr-HR" => esc_html__( 'Croatian', 'magicai-wp'),
                "cs-CZ" => esc_html__( 'Czech', 'magicai-wp'),
                "da-DK" => esc_html__( 'Danish', 'magicai-wp'),
                "nl-NL" => esc_html__( 'Dutch', 'magicai-wp'),
                "en-US" => esc_html__( 'English', 'magicai-wp'),
                "et-EE" => esc_html__( 'Estonian', 'magicai-wp'),
                "fi-FI" => esc_html__( 'Finnish', 'magicai-wp'),
                "fr-FR" => esc_html__( 'French', 'magicai-wp'),
                "gl-ES" => esc_html__( 'Galician', 'magicai-wp'),
                "de-DE" => esc_html__( 'German', 'magicai-wp'),
                "el-GR" => esc_html__( 'Greek', 'magicai-wp'),
                "he-IL" => esc_html__( 'Hebrew', 'magicai-wp'),
                "hi-IN" => esc_html__( 'Hindi', 'magicai-wp'),
                "hu-HU" => esc_html__( 'Hungarian', 'magicai-wp'),
                "is-IS" => esc_html__( 'Icelandic', 'magicai-wp'),
                "id-ID" => esc_html__( 'Indonesian', 'magicai-wp'),
                "it-IT" => esc_html__( 'Italian', 'magicai-wp'),
                "ja-JP" => esc_html__( 'Japanese', 'magicai-wp'),
                "kn-IN" => esc_html__( 'Kannada', 'magicai-wp'),
                "kk-KZ" => esc_html__( 'Kazakh', 'magicai-wp'),
                "ko-KR" => esc_html__( 'Korean', 'magicai-wp'),
                "lv-LV" => esc_html__( 'Latvian', 'magicai-wp'),
                "lt-LT" => esc_html__( 'Lithuanian', 'magicai-wp'),
                "mk-MK" => esc_html__( 'Macedonian', 'magicai-wp'),
                "ms-MY" => esc_html__( 'Malay', 'magicai-wp'),
                "mr-IN" => esc_html__( 'Marathi', 'magicai-wp'),
                "mi-NZ" => esc_html__( 'Maori', 'magicai-wp'),
                "ne-NP" => esc_html__( 'Nepali', 'magicai-wp'),
                "no-NO" => esc_html__( 'Norwegian', 'magicai-wp'),
                "fa-IR" => esc_html__( 'Persian', 'magicai-wp'),
                "pl-PL" => esc_html__( 'Polish', 'magicai-wp'),
                "pt-PT" => esc_html__( 'Portuguese', 'magicai-wp'),
                "ro-RO" => esc_html__( 'Romanian', 'magicai-wp'),
                "ru-RU" => esc_html__( 'Russian', 'magicai-wp'),
                "sr-RS" => esc_html__( 'Serbian', 'magicai-wp'),
                "sk-SK" => esc_html__( 'Slovak', 'magicai-wp'),
                "sl-SI" => esc_html__( 'Slovenian', 'magicai-wp'),
                "es-ES" => esc_html__( 'Spanish', 'magicai-wp'),
                "sw-KE" => esc_html__( 'Swahili', 'magicai-wp'),
                "sv-SE" => esc_html__( 'Swedish', 'magicai-wp'),
                "tl-PH" => esc_html__( 'Tagalog', 'magicai-wp'),
                "ta-IN" => esc_html__( 'Tamil', 'magicai-wp'),
                "th-TH" => esc_html__( 'Thai', 'magicai-wp'),
                "tr-TR" => esc_html__( 'Turkish', 'magicai-wp'),
                "uk-UA" => esc_html__( 'Ukrainian', 'magicai-wp'),
                "ur-PK" => esc_html__( 'Urdu', 'magicai-wp'),
                "vi-VN" => esc_html__( 'Vietnamese', 'magicai-wp'),
                "cy-GB" => esc_html__( 'Welsh', 'magicai-wp'),
            ],

            'GOOGLE_TTS_LANGUAGES' => [
                'af-ZA' => esc_html__('Afrikaans (South Africa)', 'magicai-wp'),
                'ar-XA' => esc_html__('Arabic', 'magicai-wp'),
                'eu-ES' => esc_html__('Basque (Spain)', 'magicai-wp'),
                'bn-IN' => esc_html__('Bengali (India)', 'magicai-wp'),
                'bg-BG' => esc_html__('Bulgarian (Bulgaria)', 'magicai-wp'),
                'ca-ES' => esc_html__('Catalan (Spain) ', 'magicai-wp'),
                'yue-HK'=> esc_html__('Chinese (Hong Kong)', 'magicai-wp'),
                'cs-CZ' => esc_html__('Czech (Czech Republic)', 'magicai-wp'),
                'da-DK' => esc_html__('Danish (Denmark)', 'magicai-wp'),
                'nl-BE' => esc_html__('Dutch (Belgium)', 'magicai-wp'),
                'nl-NL' => esc_html__('Dutch (Netherlands)', 'magicai-wp'),
                'en-AU' => esc_html__('English (Australia)', 'magicai-wp'),
                'en-IN' => esc_html__('English (India)', 'magicai-wp'),
                'en-GB' => esc_html__('English (UK)', 'magicai-wp'),
                'en-US' => esc_html__('English (US)', 'magicai-wp'),
                'fil-PH'=> esc_html__('Filipino (Philippines)', 'magicai-wp'),
                'fi-FI' => esc_html__('Finnish (Finland)', 'magicai-wp'),
                'fr-CA' => esc_html__('French (Canada)', 'magicai-wp'),
                'fr-FR' => esc_html__('French (France)', 'magicai-wp'),
                'gl-ES' => esc_html__('Galician (Spain)', 'magicai-wp'),
                'de-DE' => esc_html__('German (Germany)', 'magicai-wp'),
                'el-GR' => esc_html__('Greek (Greece)', 'magicai-wp'),
                'gu-IN' => esc_html__('Gujarati (India)', 'magicai-wp'),
                'he-IL' => esc_html__('Hebrew (Israel)', 'magicai-wp'),
                'hi-IN' => esc_html__('Hindi (India)', 'magicai-wp'),
                'hu-HU' => esc_html__('Hungarian (Hungary)', 'magicai-wp'),
                'is-IS' => esc_html__('Icelandic (Iceland)', 'magicai-wp'),
                'id-ID' => esc_html__('Indonesian (Indonesia)', 'magicai-wp'),
                'it-IT' => esc_html__('Italian (Italy)', 'magicai-wp'),
                'ja-JP' => esc_html__('Japanese (Japan)', 'magicai-wp'),
                'kn-IN' => esc_html__('Kannada (India)', 'magicai-wp'),
                'ko-KR' => esc_html__('Korean (South Korea)', 'magicai-wp'),
                'lv-LV' => esc_html__('Latvian (Latvia)', 'magicai-wp'),
                'ms-MY' => esc_html__('Malay (Malaysia)', 'magicai-wp'),
                'ml-IN' => esc_html__('Malayalam (India)', 'magicai-wp'),
                'cmn-CN'=> esc_html__('Mandarin Chinese', 'magicai-wp'),
                'cmn-TW'=> esc_html__('Mandarin Chinese (T)', 'magicai-wp'),
                'mr-IN' => esc_html__('Marathi (India)', 'magicai-wp'),
                'nb-NO' => esc_html__('Norwegian (Norway)', 'magicai-wp'),
                'pl-PL' => esc_html__('Polish (Poland)', 'magicai-wp'),
                'pt-BR' => esc_html__('Portuguese (Brazil)', 'magicai-wp'),
                'pt-PT' => esc_html__('Portuguese (Portugal)', 'magicai-wp'),
                'pa-IN' => esc_html__('Punjabi (India)', 'magicai-wp'),
                'ro-RO' => esc_html__('Romanian (Romania)', 'magicai-wp'),
                'ru-RU' => esc_html__('Russian (Russia)', 'magicai-wp'),
                'sr-RS' => esc_html__('Serbian (Cyrillic)', 'magicai-wp'),
                'sk-SK' => esc_html__('Slovak (Slovakia)', 'magicai-wp'),
                'es-ES' => esc_html__('Spanish (Spain)', 'magicai-wp'),
                'es-US' => esc_html__('Spanish (US)', 'magicai-wp'),
                'sv-SE' => esc_html__('Swedish (Sweden)', 'magicai-wp'),
                'ta-IN' => esc_html__('Tamil (India)', 'magicai-wp'),
                'te-IN' => esc_html__('Telugu (India)', 'magicai-wp'),
                'th-TH' => esc_html__('Thai (Thailand)', 'magicai-wp'),
                'tr-TR' => esc_html__('Turkish (Turkey)', 'magicai-wp'),
                'uk-UA' => esc_html__('Ukrainian (Ukraine)', 'magicai-wp'),
                'vi-VN' => esc_html__('Vietnamese (Vietnam)', 'magicai-wp'),
            ],
        
            'GOOGLE_TTS_VOICES' => [
                "af-ZA-Standard-A" =>  esc_html__("Ayanda (Female)", "magicai-wp" ),
                "ar-XA-Standard-A" =>  esc_html__("Fatima (Female)", "magicai-wp" ),
                "ar-XA-Standard-B" =>  esc_html__("Ahmed (Male)", "magicai-wp" ),
                "ar-XA-Standard-C" =>  esc_html__("Mohammed (Male)", "magicai-wp" ),
                "ar-XA-Standard-D" =>  esc_html__("Aisha (Female)", "magicai-wp" ),
                "ar-XA-Wavenet-A" =>  esc_html__("Layla (Female)", "magicai-wp" ),
                "ar-XA-Wavenet-B" =>  esc_html__("Ali (Male)", "magicai-wp" ),
                "ar-XA-Wavenet-C" =>  esc_html__("Omar (Male)", "magicai-wp" ),
                "ar-XA-Wavenet-D" =>  esc_html__("Zahra (Female)", "magicai-wp" ),
                "eu-ES-Standard-A" =>  esc_html__("Ane (Female)", "magicai-wp" ),
                "bn-IN-Standard-A" =>  esc_html__("Ananya (Female)", "magicai-wp" ),
                "bn-IN-Standard-B" =>  esc_html__("Aryan (Male)", "magicai-wp" ),
                "bn-IN-Wavenet-A" =>  esc_html__("Ishita (Female)", "magicai-wp" ),
                "bn-IN-Wavenet-B" =>  esc_html__("Arry (Male)", "magicai-wp" ),
                "bg-BG-Standard-A" =>  esc_html__("Elena (Female)", "magicai-wp" ),
                "ca-ES-Standard-A" =>  esc_html__("Laia (Female)", "magicai-wp" ),
                "yue-HK-Standard-A" =>  esc_html__("Wing (Female)", "magicai-wp" ),
                "yue-HK-Standard-B" =>  esc_html__("Ho (Male)", "magicai-wp" ),
                "yue-HK-Standard-C" =>  esc_html__("Siu (Female)", "magicai-wp" ),
                "yue-HK-Standard-D" =>  esc_html__("Lau (Male)", "magicai-wp" ),
                "cs-CZ-Standard-A" =>  esc_html__("Tereza (Female)", "magicai-wp" ),
                "cs-CZ-Wavenet-A" =>  esc_html__("Karolína (Female)", "magicai-wp" ),
                "da-DK-Standard-A" =>  esc_html__("Emma (Female)", "magicai-wp" ),
                "da-DK-Standard-A" =>  esc_html__("Freja (Female)", "magicai-wp" ),
                "da-DK-Standard-A" =>  esc_html__("Ida (Female)", "magicai-wp" ),
                "da-DK-Standard-C" =>  esc_html__("Noah (Male)", "magicai-wp" ),
                "da-DK-Standard-D" =>  esc_html__("Mathilde (Female)", "magicai-wp" ),
                "da-DK-Standard-E" =>  esc_html__("Clara (Female)", "magicai-wp" ),
                "da-DK-Wavenet-A" =>  esc_html__("Isabella (Female)", "magicai-wp" ),
                "da-DK-Wavenet-C" =>  esc_html__("Lucas (Male)", "magicai-wp" ),
                "da-DK-Wavenet-D" =>  esc_html__("Olivia (Female)", "magicai-wp" ),
                "da-DK-Wavenet-E" =>  esc_html__("Emily (Female)", "magicai-wp" ),
                "nl-BE-Standard-A" =>  esc_html__("Emma (Female)", "magicai-wp" ),
                "nl-BE-Standard-B" =>  esc_html__("Thomas (Male)", "magicai-wp" ),
                "nl-BE-Wavenet-A" =>  esc_html__("Sophie (Female)", "magicai-wp" ),
                "nl-BE-Wavenet-B" =>  esc_html__("Lucas (Male)", "magicai-wp" ),
                "nl-NL-Standard-A" =>  esc_html__("Emma (Female)", "magicai-wp" ),
                "nl-NL-Standard-B" =>  esc_html__("Daan (Male)", "magicai-wp" ),
                "nl-NL-Standard-C" =>  esc_html__("Luuk (Male)", "magicai-wp" ),
                "nl-NL-Standard-D" =>  esc_html__("Lotte (Female)", "magicai-wp" ),
                "nl-NL-Standard-E" =>  esc_html__("Sophie (Female)", "magicai-wp" ),
                "nl-NL-Wavenet-A" =>  esc_html__("Mila (Female)", "magicai-wp" ),
                "nl-NL-Wavenet-B" =>  esc_html__("Sem (Male)", "magicai-wp"),
                "nl-NL-Wavenet-C" =>  esc_html__("Stijn (Male)", "magicai-wp"),
                "nl-NL-Wavenet-D" =>  esc_html__("Fenna (Female)", "magicai-wp"),
                "nl-NL-Wavenet-E" =>  esc_html__("Eva (Female)", "magicai-wp"),
                "en-AU-News-E" =>  esc_html__("Emma (Female)", "magicai-wp"),
                "en-AU-News-F" =>  esc_html__("Olivia (Female)", "magicai-wp"),
                "en-AU-News-G" =>  esc_html__("Liam (Male)", "magicai-wp"),
                "en-AU-Standard-A" =>  esc_html__("Charlotte (Female)", "magicai-wp"),
                "en-AU-Standard-B" =>  esc_html__("Oliver (Male)", "magicai-wp"),
                "en-AU-Standard-C" =>  esc_html__("Ava (Female)", "magicai-wp"),
                "en-AU-Standard-D" =>  esc_html__("Jack (Male)", "magicai-wp"),
                "en-AU-Wavenet-A" =>  esc_html__("Sophie (Female)", "magicai-wp"),
                "en-AU-Wavenet-B" =>  esc_html__("William (Male)", "magicai-wp"),
                "en-AU-Wavenet-C" =>  esc_html__("Amelia (Female)", "magicai-wp"),
                "en-AU-Wavenet-D" =>  esc_html__("Thomas (Male)", "magicai-wp"),
                "en-IN-Standard-A" =>  esc_html__("Aditi (Female)", "magicai-wp"),
                "en-IN-Standard-B" =>  esc_html__("Arjun (Male)", "magicai-wp"),
                "en-IN-Standard-C" =>  esc_html__("Rohan (Male)", "magicai-wp"),
                "en-IN-Standard-D" =>  esc_html__("Ananya (Female)", "magicai-wp"),
                "en-IN-Wavenet-A" =>  esc_html__("Alisha (Female)", "magicai-wp"),
                "en-IN-Wavenet-B" =>  esc_html__("Aryan (Male)", "magicai-wp"),
                "en-IN-Wavenet-C" =>  esc_html__("Kabir (Male)", "magicai-wp"),
                "en-IN-Wavenet-D" =>  esc_html__("Diya (Female)", "magicai-wp"),
                "en-GB-News-G" => esc_html__("Amelia (Female)", "magicai-wp"),
                "en-GB-News-H" => esc_html__("Elise (Female)", "magicai-wp"),
                "en-GB-News-I" => esc_html__("Isabella (Female)", "magicai-wp"),
                "en-GB-News-J" => esc_html__("Jessica (Female)", "magicai-wp"),
                "en-GB-News-K" => esc_html__("Alexander (Male)", "magicai-wp"),
                "en-GB-News-L" => esc_html__("Benjamin (Male)", "magicai-wp"),
                "en-GB-News-M" => esc_html__("Charles (Male)", "magicai-wp"),
                "en-GB-Standard-A" => esc_html__("Emily (Female)", "magicai-wp"),
                "en-GB-Standard-B" => esc_html__("John (Male)", "magicai-wp"),
                "en-GB-Standard-C" => esc_html__("Mary (Female)", "magicai-wp"),
                "en-GB-Standard-D" => esc_html__("Peter (Male)", "magicai-wp"),
                "en-GB-Standard-F" => esc_html__("Sarah (Female)", "magicai-wp"),
                "en-GB-Wavenet-A" => esc_html__("Ava (Female)", "magicai-wp"),
                "en-GB-Wavenet-B" => esc_html__("David (Male)", "magicai-wp"),
                "en-GB-Wavenet-C" => esc_html__("Emily (Female)", "magicai-wp"),
                "en-GB-Wavenet-D" => esc_html__("James (Male)", "magicai-wp"),
                "en-GB-Wavenet-F" => esc_html__("Sophie (Female)", "magicai-wp"),
                "en-US-News-K" => esc_html__("Lily (Female)", "magicai-wp"),
                "en-US-News-L" => esc_html__("Olivia (Female)", "magicai-wp"),
                "en-US-News-M" => esc_html__("Noah (Male)", "magicai-wp"),
                "en-US-News-N" => esc_html__("Oliver (Male)", "magicai-wp"),
                "en-US-Standard-A" => esc_html__("Michael (Male)", "magicai-wp"),
                "en-US-Standard-B" => esc_html__("David (Male)", "magicai-wp"),
                "en-US-Standard-C" => esc_html__("Emma (Female)", "magicai-wp"),
                "en-US-Standard-D" => esc_html__("William (Male)", "magicai-wp"),
                "en-US-Standard-E" => esc_html__("Ava (Female)", "magicai-wp"),
                "en-US-Standard-F" => esc_html__("Sophia (Female)", "magicai-wp"),
                "en-US-Standard-G" => esc_html__("Isabella (Female)", "magicai-wp"),
                "en-US-Standard-H" => esc_html__("Charlotte (Female)", "magicai-wp"),
                "en-US-Standard-I" => esc_html__("James (Male)", "magicai-wp"),
                "en-US-Standard-J" => esc_html__("Lucas (Male)", "magicai-wp"),
                "en-US-Studio-M" => esc_html__("Benjamin (Male)", "magicai-wp"),
                "en-US-Studio-O" => esc_html__("Eleanor (Female)", "magicai-wp"),
                "en-US-Wavenet-A" => esc_html__("Alexander (Male)", "magicai-wp"),
                "en-US-Wavenet-B" => esc_html__("Benjamin (Male)", "magicai-wp"),
                "en-US-Wavenet-C" => esc_html__("Emily (Female)", "magicai-wp"),
                "en-US-Wavenet-D" => esc_html__("James (Male)", "magicai-wp"),
                "en-US-Wavenet-E" => esc_html__("Ava (Female)", "magicai-wp"),
                "en-US-Wavenet-F" => esc_html__("Sophia (Female)", "magicai-wp"),
                "en-US-Wavenet-G" => esc_html__("Isabella (Female)", "magicai-wp"),
                "en-US-Wavenet-H" => esc_html__("Charlotte (Female)", "magicai-wp"),
                "en-US-Wavenet-I" => esc_html__("Alexander (Male)", "magicai-wp"),
                "en-US-Wavenet-J" => esc_html__("Lucas (Male)", "magicai-wp"),
                "fil-PH-Standard-A" => esc_html__("Maria (Female)", "magicai-wp"),
                "fil-PH-Standard-B" => esc_html__("Juana (Female)", "magicai-wp"),
                "fil-PH-Standard-C" => esc_html__("Juan (Male)", "magicai-wp"),
                "fil-PH-Standard-D" => esc_html__("Pedro (Male)", "magicai-wp"),
                "fil-PH-Wavenet-A" => esc_html__("Maria (Female)", "magicai-wp"),
                "fil-PH-Wavenet-B" => esc_html__("Juana (Female)", "magicai-wp"),
                "fil-PH-Wavenet-C" => esc_html__("Juan (Male)", "magicai-wp"),
                "fil-PH-Wavenet-D" => esc_html__("Pedro (Male)", "magicai-wp"),
                "fi-FI-Standard-A" => esc_html__("Sofia (Female)", "magicai-wp"),
                "fi-FI-Wavenet-A" => esc_html__("Sofianna (Female)", "magicai-wp"),
                "fr-CA-Standard-A" => esc_html__("Emma (Female)", "magicai-wp"),
                "fr-CA-Standard-B" => esc_html__("Jean (Male)", "magicai-wp"),
                "fr-CA-Standard-C" => esc_html__("Gabrielle (Female)", "magicai-wp"),
                "fr-CA-Standard-D" => esc_html__("Thomas (Male)", "magicai-wp"),
                "fr-CA-Wavenet-A" => esc_html__("Amelie (Female)", "magicai-wp"),
                "fr-CA-Wavenet-B" => esc_html__("Antoine (Male)", "magicai-wp"),
                "fr-CA-Wavenet-C" => esc_html__("Gabrielle (Female)", "magicai-wp"),
                "fr-CA-Wavenet-D" => esc_html__("Thomas (Male)", "magicai-wp"),
                "fr-FR-Standard-A" => esc_html__("Marie (Female)", "magicai-wp"),
                "fr-FR-Standard-B" => esc_html__("Pierre (Male)", "magicai-wp"),
                "fr-FR-Standard-C" => esc_html__("Sophie (Female)", "magicai-wp"),
                "fr-FR-Standard-D" => esc_html__("Paul (Male)", "magicai-wp"),
                "fr-FR-Standard-E" => esc_html__("Julie (Female)", "magicai-wp"),
                "fr-FR-Wavenet-A" => esc_html__("Elise (Female)", "magicai-wp"),
                "fr-FR-Wavenet-B" => esc_html__("Nicolas (Male)", "magicai-wp"),
                "fr-FR-Wavenet-C" => esc_html__("Clara (Female)", "magicai-wp"),
                "fr-FR-Wavenet-D" => esc_html__("Antoine (Male)", "magicai-wp"),
                "fr-FR-Wavenet-E" => esc_html__("Amelie (Female)", "magicai-wp"),
                "gl-ES-Standard-A" => esc_html__("Ana (Female)", "magicai-wp"),
                "de-DE-Standard-A" => esc_html__("Anna (Female)", "magicai-wp"),
                "de-DE-Standard-B" => esc_html__("Max (Male)", "magicai-wp"),
                "de-DE-Standard-C" => esc_html__("Sophia (Female)", "magicai-wp"),
                "de-DE-Standard-D" => esc_html__("Paul (Male)", "magicai-wp"),
                "de-DE-Standard-E" => esc_html__("Erik (Male)", "magicai-wp"),
                "de-DE-Standard-F" => esc_html__("Lina (Female)", "magicai-wp"),
                "de-DE-Wavenet-A" => esc_html__("Eva (Female)", "magicai-wp"),
                "de-DE-Wavenet-B" => esc_html__("Felix (Male)", "magicai-wp"),
                "de-DE-Wavenet-C" => esc_html__("Emma (Female)", "magicai-wp"),
                "de-DE-Wavenet-D" => esc_html__("Lukas (Male)", "magicai-wp"),
                "de-DE-Wavenet-E" => esc_html__("Nico (Male)", "magicai-wp"),
                "de-DE-Wavenet-F" => esc_html__("Mia (Female)", "magicai-wp"),
                "el-GR-Standard-A" => esc_html__("Ελένη (Female)", "magicai-wp"),
                "el-GR-Wavenet-A" => esc_html__("Ελένη (Female)", "magicai-wp"),
                "gu-IN-Standard-A" => esc_html__("દિવ્યા (Female)", "magicai-wp"),
                "gu-IN-Standard-B" => esc_html__("કિશોર (Male)", "magicai-wp"),
                "gu-IN-Wavenet-A" => esc_html__("દિવ્યા (Female)", "magicai-wp"),
                "gu-IN-Wavenet-B" => esc_html__("કિશોર (Male)", "magicai-wp"),
                "he-IL-Standard-A" => esc_html__("Tamar (Female)", "magicai-wp"),
                "he-IL-Standard-B" => esc_html__("David (Male)", "magicai-wp"),
                "he-IL-Standard-C" => esc_html__("Michal (Female)", "magicai-wp"),
                "he-IL-Standard-D" => esc_html__("Jonathan (Male)", "magicai-wp"),
                "he-IL-Wavenet-A" => esc_html__("Yael (Female)", "magicai-wp"),
                "he-IL-Wavenet-B" => esc_html__("Eli (Male)", "magicai-wp"),
                "he-IL-Wavenet-C" => esc_html__("Abigail (Female)", "magicai-wp"),
                "he-IL-Wavenet-D" => esc_html__("Alex (Male)", "magicai-wp"),
                "hi-IN-Standard-A" => esc_html__("Aditi (Female)", "magicai-wp"),
                "hi-IN-Standard-B" => esc_html__("Abhishek (Male)", "magicai-wp"),
                "hi-IN-Standard-C" => esc_html__("Aditya (Male)", "magicai-wp"),
                "hi-IN-Standard-D" => esc_html__("Anjali (Female)", "magicai-wp"),
                "hi-IN-Wavenet-A" => esc_html__("Kiara (Female)", "magicai-wp"),
                "hi-IN-Wavenet-B" => esc_html__("Rohan (Male)", "magicai-wp"),
                "hi-IN-Wavenet-C" => esc_html__("Rishabh (Male)", "magicai-wp"),
                "hi-IN-Wavenet-D" => esc_html__("Srishti (Female)", "magicai-wp"),
                "hu-HU-Standard-A" => esc_html__("Eszter (Female)", "magicai-wp"),
                "hu-HU-Wavenet-A" => esc_html__("Lilla (Female)", "magicai-wp"),
                "is-IS-Standard-A" => esc_html__("Guðrún (Female)", "magicai-wp"),
                "id-ID-Standard-A" => esc_html__("Amelia (Female)", "magicai-wp"),
                "id-ID-Standard-B" => esc_html__("Fajar (Male)", "magicai-wp"),
                "id-ID-Standard-C" => esc_html__("Galih (Male)", "magicai-wp"),
                "id-ID-Standard-D" => esc_html__("Kiara (Female)", "magicai-wp"),
                "id-ID-Wavenet-A" => esc_html__("Nadia (Female)", "magicai-wp"),
                "id-ID-Wavenet-B" => esc_html__("Reza (Male)", "magicai-wp"),
                "id-ID-Wavenet-C" => esc_html__("Satria (Male)", "magicai-wp"),
                "id-ID-Wavenet-D" => esc_html__("Vania (Female)", "magicai-wp"),
                "it-IT-Standard-A" => esc_html__("Chiara (Female)", "magicai-wp"),
                "it-IT-Standard-B" => esc_html__("Elisa (Female)", "magicai-wp"),
                "it-IT-Standard-C" => esc_html__("Matteo (Male)", "magicai-wp"),
                "it-IT-Standard-D" => esc_html__("Riccardo (Male)", "magicai-wp"),
                "it-IT-Wavenet-A" => esc_html__("Valentina (Female)", "magicai-wp"),
                "it-IT-Wavenet-B" => esc_html__("Vittoria (Female)", "magicai-wp"),
                "it-IT-Wavenet-C" => esc_html__("Andrea (Male)", "magicai-wp"),
                "it-IT-Wavenet-D" => esc_html__("Luca (Male)", "magicai-wp"),
                "ja-JP-Standard-A" => esc_html__("Akane (Female)", "magicai-wp"),
                "ja-JP-Standard-B" => esc_html__("Emi (Female)", "magicai-wp"),
                "ja-JP-Standard-C" => esc_html__("Daisuke (Male)", "magicai-wp"),
                "ja-JP-Standard-D" => esc_html__("Kento (Male)", "magicai-wp"),
                "ja-JP-Wavenet-A" => esc_html__("Haruka (Female)", "magicai-wp"),
                "ja-JP-Wavenet-B" => esc_html__("Rin (Female)", "magicai-wp"),
                "ja-JP-Wavenet-C" => esc_html__("Shun (Male)", "magicai-wp"),
                "ja-JP-Wavenet-D" => esc_html__("Yuta (Male)", "magicai-wp"),
                "kn-IN-Standard-A" => esc_html__("Dhanya (Female)", "magicai-wp"),
                "kn-IN-Standard-B" => esc_html__("Keerthi (Male)", "magicai-wp"),
                "kn-IN-Wavenet-A" => esc_html__("Meena (Female)", "magicai-wp"),
                "kn-IN-Wavenet-B" => esc_html__("Nandini (Male)", "magicai-wp"),
                "ko-KR-Standard-A" => esc_html__("So-young (Female)", "magicai-wp"),
                "ko-KR-Standard-B" => esc_html__("Se-yeon (Female)", "magicai-wp"),
                "ko-KR-Standard-C" => esc_html__("Min-soo (Male)", "magicai-wp"),
                "ko-KR-Standard-D" => esc_html__("Seung-woo (Male)", "magicai-wp"),
                "ko-KR-Wavenet-A" => esc_html__("Ji-soo (Female)", "magicai-wp"),
                "ko-KR-Wavenet-B" => esc_html__("Yoon-a (Female)", "magicai-wp"),
                "ko-KR-Wavenet-C" => esc_html__("Tae-hyun (Male)", "magicai-wp"),
                "ko-KR-Wavenet-D" => esc_html__("Jun-ho (Male)", "magicai-wp"),
                "lv-LV-Standard-A" => esc_html__("Raivis (Male)", "magicai-wp"),
                "lv-LT-Standard-A" =>  esc_html__("Raivis (Male)", "magicai-wp"),
                "ms-MY-Standard-A" => esc_html__("Amira (Female)", "magicai-wp"),
                "ms-MY-Standard-B" => esc_html__("Danial (Male)", "magicai-wp"),
                "ms-MY-Standard-C" => esc_html__("Eira (Female)", "magicai-wp"),
                "ms-MY-Standard-D" => esc_html__("Farhan (Male)", "magicai-wp"),
                "ms-MY-Wavenet-A" => esc_html__("Hana (Female)", "magicai-wp"),
                "ms-MY-Wavenet-B" => esc_html__("Irfan (Male)", "magicai-wp"),
                "ms-MY-Wavenet-C" => esc_html__("Janna (Female)", "magicai-wp"),
                "ms-MY-Wavenet-D" => esc_html__("Khairul (Male)", "magicai-wp"),
                "ml-IN-Standard-A" => esc_html__("Aishwarya (Female)", "magicai-wp"),
                "ml-IN-Standard-B" => esc_html__("Dhruv (Male)", "magicai-wp"),
                "ml-IN-Wavenet-A" => esc_html__("Deepthi (Female)", "magicai-wp"),
                "ml-IN-Wavenet-B" => esc_html__("Gautam (Male)", "magicai-wp"),
                "ml-IN-Wavenet-C" => esc_html__("Isha (Female)", "magicai-wp"),
                "ml-IN-Wavenet-D" => esc_html__("Kabir (Male)", "magicai-wp"),
                "cmn-CN-Standard-A" => esc_html__("Xiaomei (Female)", "magicai-wp"),
                "cmn-CN-Standard-B" => esc_html__("Lijun (Male)", "magicai-wp"),
                "cmn-CN-Standard-C" => esc_html__("Minghao (Male)", "magicai-wp"),
                "cmn-CN-Standard-D" => esc_html__("Yingying (Female)", "magicai-wp"),
                "cmn-CN-Wavenet-A" => esc_html__("Shanshan (Female)", "magicai-wp"),
                "cmn-CN-Wavenet-B" => esc_html__("Chenchen (Male)", "magicai-wp"),
                "cmn-CN-Wavenet-C" => esc_html__("Jiahao (Male)", "magicai-wp"),
                "cmn-CN-Wavenet-D" => esc_html__("Yueyu (Female)", "magicai-wp"),
                "cmn-TW-Standard-A" => esc_html__("Jingwen (Female)", "magicai-wp"),
                "cmn-TW-Standard-B" => esc_html__("Jinghao (Male)", "magicai-wp"),
                "cmn-TW-Standard-C" => esc_html__("Tingting (Female)", "magicai-wp"),
                "cmn-TW-Wavenet-A" => esc_html__("Yunyun (Female)", "magicai-wp"),
                "cmn-TW-Wavenet-B" => esc_html__("Zhenghao (Male)", "magicai-wp"),
                "cmn-TW-Wavenet-C" => esc_html__("Yuehan (Female)", "magicai-wp"),
                "mr-IN-Standard-A" => esc_html__("Anjali (Female)", "magicai-wp"),
                "mr-IN-Standard-B" => esc_html__("Aditya (Male)", "magicai-wp"),
                "mr-IN-Standard-C" => esc_html__("Dipti (Female)", "magicai-wp"),
                "mr-IN-Wavenet-A" => esc_html__("Gauri (Female)", "magicai-wp"),
                "mr-IN-Wavenet-B" => esc_html__("Harsh (Male)", "magicai-wp"),
                "mr-IN-Wavenet-C" => esc_html__("Ishita (Female)", "magicai-wp"),
                "nb-NO-Standard-A" => esc_html__("Ingrid (Female)", "magicai-wp"),
                "nb-NO-Standard-B" => esc_html__("Jonas (Male)", "magicai-wp"),
                "nb-NO-Standard-C" => esc_html__("Marit (Female)", "magicai-wp"),
                "nb-NO-Standard-D" => esc_html__("Olav (Male)", "magicai-wp"),
                "nb-NO-Standard-E" => esc_html__("Silje (Female)", "magicai-wp"),
                "nb-NO-Wavenet-A" => esc_html__("Astrid (Female)", "magicai-wp"),
                "nb-NO-Wavenet-B" => esc_html__("Eirik (Male)", "magicai-wp"),
                "nb-NO-Wavenet-C" => esc_html__("Inger (Female)", "magicai-wp"),
                "nb-NO-Wavenet-D" => esc_html__("Kristian (Male)", "magicai-wp"),
                "nb-NO-Wavenet-E" => esc_html__("Trine (Female)", "magicai-wp"),
                "pl-PL-Standard-A" => esc_html__("Agata (Female)", "magicai-wp"),
                "pl-PL-Standard-B" => esc_html__("Bartosz (Male)", "magicai-wp"),
                "pl-PL-Standard-C" => esc_html__("Kamil (Male)", "magicai-wp"),
                "pl-PL-Standard-D" => esc_html__("Julia (Female)", "magicai-wp"),
                "pl-PL-Standard-E" => esc_html__("Magdalena (Female)", "magicai-wp"),
                "pl-PL-Wavenet-A" => esc_html__("Natalia (Female)", "magicai-wp"),
                "pl-PL-Wavenet-B" => esc_html__("Paweł (Male)", "magicai-wp"),
                "pl-PL-Wavenet-C" => esc_html__("Tomasz (Male)", "magicai-wp"),
                "pl-PL-Wavenet-D" => esc_html__("Zofia (Female)", "magicai-wp"),
                "pl-PL-Wavenet-E" => esc_html__("Wiktoria (Female)", "magicai-wp"),
                "pt-BR-Standard-A" => esc_html__("Ana (Female)", "magicai-wp"),
                "pt-BR-Standard-B" => esc_html__("Carlos (Male)", "magicai-wp"),
                "pt-BR-Standard-C" => esc_html__("Maria (Female)", "magicai-wp"),
                "pt-BR-Wavenet-A" => esc_html__("Julia (Female)", "magicai-wp"),
                "pt-BR-Wavenet-B" => esc_html__("João (Male)", "magicai-wp"),
                "pt-BR-Wavenet-C" => esc_html__("Fernanda (Female)", "magicai-wp"),
                "pt-PT-Standard-A" => esc_html__("Maria (Female)", "magicai-wp"),
                "pt-PT-Standard-B" => esc_html__("José (Male)", "magicai-wp"),
                "pt-PT-Standard-C" => esc_html__("Luís (Male)", "magicai-wp"),
                "pt-PT-Standard-D" => esc_html__("Ana (Female)", "magicai-wp"),
                "pt-PT-Wavenet-A" => esc_html__("Catarina (Female)", "magicai-wp"),
                "pt-PT-Wavenet-B" => esc_html__("Miguel (Male)", "magicai-wp"),
                "pt-PT-Wavenet-C" => esc_html__("João (Male)", "magicai-wp"),
                "pt-PT-Wavenet-D" => esc_html__("Marta (Female)", "magicai-wp"),
                "pa-IN-Standard-A" => esc_html__("Harpreet (Female)", "magicai-wp"),
                "pa-IN-Standard-B" => esc_html__("Gurpreet (Male)", "magicai-wp"),
                "pa-IN-Standard-C" => esc_html__("Jasmine (Female)", "magicai-wp"),
                "pa-IN-Standard-D" => esc_html__("Rahul (Male)", "magicai-wp"),
                "pa-IN-Wavenet-A" => esc_html__("Simran (Female)", "magicai-wp"),
                "pa-IN-Wavenet-B" => esc_html__("Amardeep (Male)", "magicai-wp"),
                "pa-IN-Wavenet-C" => esc_html__("Kiran (Female)", "magicai-wp"),
                "pa-IN-Wavenet-D" => esc_html__("Raj (Male)", "magicai-wp"),
                "ro-RO-Standard-A" => esc_html__("Maria (Female)", "magicai-wp"),
                "ro-RO-Wavenet-A" => esc_html__("Ioana (Female)", "magicai-wp"),
                "ru-RU-Standard-A" => esc_html__("Anastasia", "magicai-wp"),
                "ru-RU-Standard-B" => esc_html__("Alexander", "magicai-wp"),
                "ru-RU-Standard-C" => esc_html__("Elizabeth", "magicai-wp"),
                "ru-RU-Standard-D" => esc_html__("Michael", "magicai-wp"),
                "ru-RU-Standard-E" => esc_html__("Victoria", "magicai-wp"),
                "ru-RU-Wavenet-A" => esc_html__("Daria", "magicai-wp"),
                "ru-RU-Wavenet-B" => esc_html__("Dmitry", "magicai-wp"),
                "ru-RU-Wavenet-C" => esc_html__("Kristina", "magicai-wp"),
                "ru-RU-Wavenet-D" => esc_html__("Ivan", "magicai-wp"),
                "ru-RU-Wavenet-E" => esc_html__("Sophia", "magicai-wp"),
                "sr-RS-Standard-A" => esc_html__("Ana", "magicai-wp"),
                "sk-SK-Standard-A" => esc_html__("Mária (Female)", "magicai-wp"),
                "sk-SK-Wavenet-A" => esc_html__("Zuzana (Female)", "magicai-wp"),
                "es-ES-Standard-A" =>  esc_html__("María (Female)", "magicai-wp"),
                "es-ES-Standard-B" =>  esc_html__("José (Male)", "magicai-wp"),
                "es-ES-Standard-C" =>  esc_html__("Ana (Female)", "magicai-wp"),
                "es-ES-Standard-D" =>  esc_html__("Isabel (Female)", "magicai-wp"),
                "es-ES-Wavenet-B" =>  esc_html__("Pedro (Male)", "magicai-wp"),
                "es-ES-Wavenet-C" =>  esc_html__("Laura (Female)", "magicai-wp"),
                "es-ES-Wavenet-D" =>  esc_html__("Julia (Female)", "magicai-wp"),
                "es-US-News-D" =>  esc_html__("Diego (Male)", "magicai-wp"),
                "es-US-News-E" =>  esc_html__("Eduardo (Male)", "magicai-wp"),
                "es-US-News-F" =>  esc_html__("Fátima (Female)", "magicai-wp"),
                "es-US-News-G" =>  esc_html__("Gabriela (Female)", "magicai-wp"),
                "es-US-Standard-A" =>  esc_html__("Ana (Female)", "magicai-wp"),
                "es-US-Standard-B" =>  esc_html__("José (Male)", "magicai-wp"),
                "es-US-Standard-C" =>  esc_html__("Carlos (Male)", "magicai-wp"),
                "es-US-Studio-B" =>  esc_html__("Miguel (Male)", "magicai-wp"),
                "es-US-Wavenet-A" =>  esc_html__("Laura (Female)", "magicai-wp"),
                "es-US-Wavenet-B" =>  esc_html__("Pedro (Male)", "magicai-wp"),
                "es-US-Wavenet-C" =>  esc_html__("Pablo (Male)", "magicai-wp"),
                "sv-SE-Standard-A" =>  esc_html__("Ebba (Female)", "magicai-wp"),
                "sv-SE-Standard-B" =>  esc_html__("Saga (Female)", "magicai-wp"),
                "sv-SE-Standard-C" =>  esc_html__("Linnea (Female)", "magicai-wp"),
                "sv-SE-Standard-D" =>  esc_html__("Erik (Male)", "magicai-wp"),
                "sv-SE-Standard-E" =>  esc_html__("Anton (Male)", "magicai-wp"),
                "sv-SE-Wavenet-A" =>  esc_html__("Astrid (Female)", "magicai-wp"),
                "sv-SE-Wavenet-B" =>  esc_html__("Elin (Female)", "magicai-wp"),
                "sv-SE-Wavenet-C" =>  esc_html__("Oskar (Male)", "magicai-wp"),
                "sv-SE-Wavenet-D" =>  esc_html__("Hanna (Female)", "magicai-wp"),
                "sv-SE-Wavenet-E" =>  esc_html__("Felix (Male)", "magicai-wp"),
                "ta-IN-Standard-A" =>  esc_html__("Anjali (Female)", "magicai-wp"),
                "ta-IN-Standard-B" =>  esc_html__("Karthik (Male)", "magicai-wp"),
                "ta-IN-Standard-C" =>  esc_html__("Priya (Female)", "magicai-wp"),
                "ta-IN-Standard-D" =>  esc_html__("Ravi (Male)", "magicai-wp"),
                "ta-IN-Wavenet-A" =>  esc_html__("Lakshmi (Female)", "magicai-wp"),
                "ta-IN-Wavenet-B" =>  esc_html__("Suresh (Male)", "magicai-wp"),
                "ta-IN-Wavenet-C" =>  esc_html__("Uma (Female)", "magicai-wp"),
                "ta-IN-Wavenet-D" =>  esc_html__("Venkatesh (Male)", "magicai-wp"),
                "-IN-Standard-A" =>  esc_html__(" - (Female)", "magicai-wp"),
                "-IN-Standard-B" =>  esc_html__(" - (Male)", "magicai-wp"),
                "th-TH-Standard-A" =>  esc_html__(" - (Female)", "magicai-wp"),
                "tr-TR-Standard-A" =>  esc_html__("Ayşe (Female)", "magicai-wp"),
                "tr-TR-Standard-B" =>  esc_html__("Berk (Male)", "magicai-wp"),
                "tr-TR-Standard-C" =>  esc_html__("Cansu (Female)", "magicai-wp"),
                "tr-TR-Standard-D" =>  esc_html__( "Deniz (Female)", "magicai-wp"),
                "tr-TR-Standard-E" =>  esc_html__( "Emre (Male)", "magicai-wp"),
                "tr-TR-Wavenet-A" =>  esc_html__( "Gül (Female)", "magicai-wp"),
                "tr-TR-Wavenet-B" =>  esc_html__( "Mert (Male)", "magicai-wp"),
                "tr-TR-Wavenet-C" =>  esc_html__( "Nilay (Female)", "magicai-wp"),
                "tr-TR-Wavenet-D" =>  esc_html__( "Selin (Female)", "magicai-wp"),
                "tr-TR-Wavenet-E" =>  esc_html__( "Tolga (Male)", "magicai-wp"),
                "uk-UA-Standard-A" =>  esc_html__( "Anya (Female)", "magicai-wp"),
                "uk-UA-Wavenet-A" =>  esc_html__( "Dasha (Female)", "magicai-wp"),
                "vi-VN-Standard-A" =>  esc_html__( "Mai (Female)", "magicai-wp"),
                "vi-VN-Standard-B" =>  esc_html__( "Nam (Male)", "magicai-wp"),
                "vi-VN-Standard-C" =>  esc_html__( "Hoa (Female)", "magicai-wp"),
                "vi-VN-Standard-D" =>  esc_html__( "Huy (Male)", "magicai-wp"),
                "vi-VN-Wavenet-A" =>  esc_html__( "Lan (Female)", "magicai-wp"),
                "vi-VN-Wavenet-B" =>  esc_html__( "Son (Male)", "magicai-wp"),
                "vi-VN-Wavenet-C" =>  esc_html__( "Thao (Female)", "magicai-wp"),
                "vi-VN-Wavenet-D" =>  esc_html__( "Tuan (Male)", "magicai-wp"),
            ],
            
        ];

        return $const[$name];

    }

    /**
     * Counts the number of words in a given text.
     *
     * This function counts the number of words in the provided text. It supports
     * counting words in multiple languages, including Chinese characters, by
     * utilizing appropriate methods based on the text's encoding.
     *
     * @since 1.3
     *
     * @param string $text The text for which word count needs to be calculated.
     * @return int The number of words in the provided text.
     */
    function get_word_count($text){


        $encoding = mb_detect_encoding($text);
    
        if ($encoding === 'UTF-8') {
            // Count Chinese words by splitting the string into individual characters
            $words = preg_match_all('/\p{Han}|\p{L}+|\p{N}+/u', $text);
        } else {
            // For other languages, use str_word_count()
            $words = str_word_count($text, 0, $encoding);
        }
    
        return intval($words);
    
    }

    /**
     * Check the status of a user's IP address.
     *
     * This function checks whether the provided IP address exists in the stored IP storage.
     * If no IP address is provided, it automatically retrieves the user's IP address.
     *
     * @since 1.4
     *
     * @param string|null $ip     Optional. The IP address to check. Default is null.
     * @param bool        $delete Optional. Whether to delete the IP address from storage if found. Default is false.
     * @return bool True if the IP address exists in storage, false otherwise.
     */
    function check_user_ip_status( $ip = null, $delete = false ) {

        // If no IP address is provided, retrieve the user's IP address.
        if ( empty( $ip ) ) {
            $ip = magicai_helper()->get_ip() ?? $_SERVER['REMOTE_ADDR'];
        }

        // Retrieve the stored IP storage.
        $ip_storage = get_option( 'magicai_ip_storage', array() );
        // Check if the provided IP address exists in the storage.
        if (($key = array_search($ip, $ip_storage)) !== false) {
            if ( $delete ) {
                unset($ip_storage[$key]);
                update_option( 'magicai_ip_storage', $ip_storage );
            }
            return true; // IP address found in storage.
        }

        return false; // IP address not found in storage.

    }

    /**
     * Checks if the current request is a frontend AJAX request.
     *
     * This function determines if the current request is an AJAX request
     * initiated from the frontend of the website.
     *
     * @return bool True if the request is a frontend AJAX request, false otherwise.
     * 
     * @since 1.4
     */
    function request_is_frontend_ajax(){
        $script_filename = isset($_SERVER['SCRIPT_FILENAME']) ? $_SERVER['SCRIPT_FILENAME'] : '';

        //Try to figure out if frontend AJAX request... If we are DOING_AJAX; let's look closer
        if((defined('DOING_AJAX') && DOING_AJAX)){
                //From wp-includes/functions.php, wp_get_referer() function.
                $ref = '';
                if ( ! empty( $_REQUEST['_wp_http_referer'] ) )
                    $ref = wp_unslash( $_REQUEST['_wp_http_referer'] );
                elseif ( ! empty( $_SERVER['HTTP_REFERER'] ) )
                    $ref = wp_unslash( $_SERVER['HTTP_REFERER'] );

            //If referer does not contain admin URL and we are using the admin-ajax.php endpoint, this is likely a frontend AJAX request
            if(((strpos($ref, admin_url()) === false) && (basename($script_filename) === 'admin-ajax.php'))) {
                return true;
            }
        }

        //If no checks triggered, we end up here - not an AJAX request.
        return false;
    }

}

/**
 * Main instance of MagicAI_Helper.
 *
 * Returns the main instance of MagicAI_Helper to prevent the need to use globals.
 *
 * @return MagicAI_Helper
 */
function magicai_helper() {
	return MagicAI_Helper::instance();
}
