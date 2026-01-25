<?php 

// Hook into the 'post_row_actions' filter to add your custom row action.
add_filter('post_row_actions', 'magicai_post_row_actions', 10, 2);

// Hooks for duplicate post
add_action( 'admin_notices', 'magicai_duplication_admin_notice' );
add_action( 'admin_action_magicai_duplicate_post', 'magicai_duplicate_post' );

// Add the custom MIME type to the list of allowed file types.
add_filter( 'upload_mimes', 'magicai_allow_audio_mimes' );
add_filter( 'wp_check_filetype_and_ext', 'magicai_fix_wp_check_filetype_and_ext', 10, 4 );

/**
 * AJAX Hooks
 *
 * These hooks define various AJAX actions in WordPress.
 * AJAX (Asynchronous JavaScript and XML) allows you to perform actions without
 * reloading the entire page, enabling more dynamic and interactive user experiences.
 * @see https://developer.wordpress.org/reference/hooks/wp_ajax_action/
 * @see https://developer.wordpress.org/reference/hooks/wp_ajax_nopriv_action/
 */
add_action('wp_ajax_fetch_post_details', 'fetch_post_details'); 
add_action('wp_ajax_nopriv_fetch_post_details', 'fetch_post_details');

add_action('wp_ajax_magicai_add_new_post_as_draft', 'add_new_post_as_draft');
add_action('wp_ajax_magicai_add_new_documents', 'add_new_documents');

add_action('wp_ajax_magicai_delete_attachment', 'magicai_delete_attachment');

add_action('wp_ajax_magicai_create_new_chat', 'magicai_create_new_chat');
add_action('wp_ajax_magicai_save_chat_data', 'magicai_save_chat_data');
add_action('wp_ajax_magicai_get_chat_data', 'magicai_get_chat_data');
add_action('wp_ajax_magicai_get_chat', 'magicai_get_chat');
add_action('wp_ajax_magicai_delete_chat', 'magicai_delete_chat');
add_action('wp_ajax_magicai_edit_chat_name', 'magicai_edit_chat_name');

add_action('wp_ajax_magicai_fetch_rss', 'magicai_fetch_rss');

add_action('wp_ajax_magicai_chatbot_get_chat', 'magicai_chatbot_get_chat');
add_action('wp_ajax_magicai_chatbot_get_chat_conversations', 'magicai_chatbot_get_chat_conversations');
add_action('wp_ajax_magicai_chatbot_get_chat_data', 'magicai_chatbot_get_chat_data');
add_action('wp_ajax_magicai_chatbot_save_chat_data', 'magicai_chatbot_save_chat_data');
add_action('wp_ajax_magicai_chatbot_start_new_chat', 'magicai_chatbot_start_new_chat');
add_action('wp_ajax_nopriv_magicai_chatbot_get_chat', 'magicai_chatbot_get_chat');
add_action('wp_ajax_nopriv_magicai_chatbot_get_chat_data', 'magicai_chatbot_get_chat_data');
add_action('wp_ajax_nopriv_magicai_chatbot_save_chat_data', 'magicai_chatbot_save_chat_data');
add_action('wp_ajax_nopriv_magicai_chatbot_start_new_chat', 'magicai_chatbot_start_new_chat');

/**
 * Add a custom row action to the WordPress post list table.
 *
 * @param array $actions An array of row action links.
 * @param WP_Post $post The current WordPress post object.
 *
 * @return array Updated array of row action links.
 */
function magicai_post_row_actions($actions, $post) {

    // check user 
    if( ! current_user_can( 'edit_posts' ) ) {
		return $actions;
	}

    // Check if the current post type is 'post'.
    if ($post->post_type == 'post' || $post->post_type == 'product') {

        $duplicate_url = wp_nonce_url(
            add_query_arg(
                array(
                    'action' => 'magicai_duplicate_post',
                    'post' => $post->ID,
                ),
                'admin.php'
            ),
            basename(__FILE__),
            'duplicate_nonce'
        );

        // Add your custom action link.
        $actions['magicai'] = sprintf( 
            '<a class="magicai-row-action" href="%1$s" data-postid="%3$s" data-duplicate-url="%4$s">%2$s</a>', 
            '#', 
            '<img width="12" src="' . MAGICAI_URL . 'assets/img/logo.svg"> MagicAI',
            $post->ID,
            $duplicate_url
        );
    }

    return $actions;
}

/**
 * WordPress AJAX action to fetch post details.
 *
 * This function retrieves the title, content, and tags of a post by its ID.
 *
 * @since 1.0.0
 */
function fetch_post_details() {
    // Check if the post ID parameter is provided in the AJAX request.
    if (isset($_POST['post_id'])) {
        $post_id = intval($_POST['post_id']); // Sanitize and get the post ID.

        // Get the post object using the post ID.
        $post = get_post($post_id);

        if ($post) {
            // Get the post title.
            $post_title = $post->post_title;

            // Get the post content.
            $post_content = $post->post_content;
            
            // Get the post tags.
            $post_tags = wp_get_post_tags($post_id);

            // Initialize an empty string for tag names.
            $tag_names = '';

            // Check if there are tags and concatenate tag names into a string.
            if ($post_tags) {
                $tag_names = implode(', ', wp_list_pluck($post_tags, 'name'));
            }

            $response = array(
                'title'   => $post_title,
                'content' => $post_content,
                'tags'    => $tag_names,
            );

            // Send the JSON response.
            wp_send_json_success($response);
        } else {
            // Post not found.
            wp_send_json_error(array('message' => 'Post not found'));
        }
    } else {
        // Invalid request, no post ID provided.
        wp_send_json_error(array('message' => 'Invalid request'));
    }

    // Make sure to exit after sending the JSON response.
    exit;
}

/**
 * Duplicate a WordPress post.
 *
 * This function duplicates an existing post and creates a new draft with the same content.
 *
 * @since 1.0.0
 */
function magicai_duplicate_post() {

    global $wpdb;

	// check if post ID has been provided and action
	if ( empty( $_GET[ 'post' ] ) ) {
		wp_die( 'No post to duplicate has been provided!' );
	}

	// Nonce verification
	if ( ! isset( $_GET[ 'duplicate_nonce' ] ) || ! wp_verify_nonce( $_GET[ 'duplicate_nonce' ], basename( __FILE__ ) ) ) {
		return;
	}

	// Get the original post id
	$post_id = absint( $_GET[ 'post' ] );

	// And all the original post data then
	$post = get_post( $post_id );

	/*
	 * if you don't want current user to be the new post author,
	 * then change next couple of lines to this: $new_post_author = $post->post_author;
	 */
	$current_user = wp_get_current_user();
	$new_post_author = $current_user->ID;

	// if post data exists (I am sure it is, but just in a case), create the post duplicate
	if ( $post ) {

		// new post data array
		$args = array(
			'comment_status' => $post->comment_status,
			'ping_status'    => $post->ping_status,
			'post_author'    => $new_post_author,
			'post_content'   => $post->post_content,
			'post_excerpt'   => $post->post_excerpt,
			'post_name'      => $post->post_name,
			'post_parent'    => $post->post_parent,
			'post_password'  => $post->post_password,
			'post_status'    => 'draft',
			'post_title'     => $post->post_title,
			'post_type'      => $post->post_type,
			'to_ping'        => $post->to_ping,
			'menu_order'     => $post->menu_order
		);

		// insert the post by wp_insert_post() function
		$new_post_id = wp_insert_post( $args );

		/*
		 * get all current post terms ad set them to the new post draft
		 */
		$taxonomies = get_object_taxonomies( get_post_type( $post ) ); // returns array of taxonomy names for post type, ex array("category", "post_tag");
		if( $taxonomies ) {
			foreach ( $taxonomies as $taxonomy ) {
				$post_terms = wp_get_object_terms( $post_id, $taxonomy, array( 'fields' => 'slugs' ) );
				wp_set_object_terms( $new_post_id, $post_terms, $taxonomy, false );
			}
		}

		// duplicate all post meta
        $post_meta_data = $wpdb->get_results( "SELECT meta_key, meta_value FROM $wpdb->postmeta WHERE post_id=$post_id" );
        if ( count( $post_meta_data ) != 0 ) {
            $clone_query = "INSERT INTO $wpdb->postmeta (post_id, meta_key, meta_value) ";
            foreach ( $post_meta_data as $meta_data ) {
                $meta_key = sanitize_text_field( $meta_data->meta_key );
                $meta_value = addslashes( $meta_data->meta_value );
                $clone_query_select[]= "SELECT $new_post_id, '$meta_key', '$meta_value'";
            }
            $clone_query.= implode( " UNION ALL ", $clone_query_select );
            $wpdb->query( $clone_query );
        }
		
        MagicAI_Logs::instance()->add_log(
            'post_quick_action',
            'Post Duplicated',
            $post->post_title,
        );

		// redirect to all posts with a message
		wp_safe_redirect(
			add_query_arg(
				array(
					'post_type' => ( 'post' !== get_post_type( $post ) ? get_post_type( $post ) : false ),
					'saved' => 'post_duplication_created' // just a custom slug here
				),
				admin_url( 'edit.php' )
			)
		);
		exit;

	} else {
		wp_die( 'Post creation failed, could not find original post.' );
	}

}

/**
 * Display an admin notice when a post duplication is created.
 *
 * This function checks the current screen and displays an admin notice when a post duplication is created.
 * @since 1.0.0
 */
function magicai_duplication_admin_notice() {

	// Get the current screen
	$screen = get_current_screen();

	if ( 'edit' !== $screen->base ) {
		return;
	}

    //Checks if settings updated
    if ( isset( $_GET[ 'saved' ] ) && 'post_duplication_created' == $_GET[ 'saved' ] ) {

		echo sprintf('<div class="notice notice-success is-dismissible"><p>%s</p></div>', esc_html__('Post copy created.', 'magicai-wp'));
		 
    }
}

/**
 * Add a new post to documents cpt.
 *
 * @since 1.0.0
 */
function add_new_documents() {

	$post_title = !empty( $_POST['title'] ) ? sanitize_text_field( $_POST['title'] ) : 'Text #' . uniqid();
	$post_content = !empty( $_POST['content'] ) ? wp_kses_post( $_POST['content'] ) : '';

	$post_id = wp_insert_post( [
		'post_type' => 'magicai-documents',
		'post_status' => 'publish',
		'post_content' => $post_content,
		'post_title' => $post_title,
		'meta_input' => [ 
			'_magicai_doc_type' => 'text',
			'_magicai_userid' => get_current_user_id()
		 ],
	] );

	if ( is_wp_error( $post_id ) ) {
		wp_send_json( [
			'error' => true,
			'message' =>  $post_id->get_error_message(),
		] );
	}

	MagicAI_Stats::instance()->add_stats( [
		'type' => 'text',
		'ai' => 'custom_generator',
		'count' => magicai_helper()->get_word_count( $post_content ),
	] );

	wp_send_json( [
		'message' => esc_html__( 'Result saved to Documents.', 'magicai-wp' ),
	] );

}

/**
 * Add a new post as a draft.
 *
 * This function takes input from a form and creates a new post as a draft.
 *
 * @since 1.0.0
 */
function add_new_post_as_draft() {

	// sanitize 
	$post_title = !empty( $_POST['post_title'] ) ? sanitize_text_field( $_POST['post_title'] ) : '';
	$post_content = !empty( $_POST['post_content'] ) ? wp_kses_post( $_POST['post_content'] ) : '';
	$post_tags = !empty( $_POST['post_tags'] ) ? sanitize_text_field( $_POST['post_tags'] ) : '';
	$post_image = !empty( $_POST['post_image'] ) ? sanitize_text_field( $_POST['post_image'] ) : '';
	$post_type = !empty( $_POST['post_type'] ) ? sanitize_text_field( $_POST['post_type'] ) : 'post';

	if ( !class_exists( 'WooCommerce' ) && $post_type == 'product' ) {
		$post_type = 'post';
	}

	$postarr = [
		'post_type' => $post_type,
		'post_title' => $post_title,
		'post_content' => $post_content,
		'post_tags' => $post_tags,
		'post_author' => get_current_user_id(),
	];

	$post_id = wp_insert_post( $postarr );

	if ( is_wp_error( $post_id ) ) {
		MagicAI_Logs::instance()->add_log(
            'error',
            'Post not Created',
            $post_id->get_error_message(),
        );

		wp_send_json_error( $post_id->get_error_message() );
	} else {
		MagicAI_Logs::instance()->add_log(
            'completion',
            'Post Saved as Draft',
            get_the_title( $post_id ),
        );

		wp_set_post_terms( $post_id, $post_tags, $post_type.'_tag', false );

		if ( $post_image ) {
			magicai_helper()->insert_image_to_post( $post_id, $post_image );
		}

		wp_send_json_success( [
			'post_id' => $post_id,
			'output' => sprintf( 
				'<a href="%s" target="_blank">%s <span>%s</span></a>', 
				get_edit_post_link( $post_id ),
				'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M10 6V8H5V19H16V14H18V20C18 20.5523 17.5523 21 17 21H4C3.44772 21 3 20.5523 3 20V7C3 6.44772 3.44772 6 4 6H10ZM21 3V11H19L18.9999 6.413L11.2071 14.2071L9.79289 12.7929L17.5849 5H13V3H21Z"></path></svg> ',
				esc_html__( 'Visit Post', 'magicai-wp' )
			)
		] );
	}

}

/**
 * Delete an attachment.
 *
 * This function deletes an attachment identified by the provided attachment ID.
 *
 * @since 1.0.0
 */
function magicai_delete_attachment() {
	$attachment_id = intval( $_POST['attachment_id'] );

	if ( ! empty( $attachment_id ) ) {
		if ( is_attachment( $attachment_id ) ) {
			wp_delete_attachment( $attachment_id );
		} else {
			MagicAI_Amazon_S3::instance()->delete_image( get_post_field( 'post_title', $attachment_id ) );
			wp_delete_post( $attachment_id );
		}
		wp_send_json_success( 'Attachment deleted!' );
	}
}

/**
 * Create a new chat post.
 *
 * This function creates a new chat post of the 'magicai-chat' post type and adds initial chat messages.
 *
 * @since 1.0.0
 */
function magicai_create_new_chat() {

	$type = !empty( $_POST['type'] ) ? sanitize_text_field( $_POST['type'] ) : 'chat';
	$type = str_replace( 'pdf', 'chat-pdf', $type );

	$post_id = wp_insert_post(
		[
			'post_type' => "magicai-$type",
			'post_status' => 'publish',
			'meta_input' => [ 
				'_magicai_userid' => get_current_user_id(),
			]
		]
	);

	if( ! is_wp_error( $post_id ) ) {
		wp_update_post(
			[
				'ID' => $post_id,
				'post_title'=> sprintf( 'Chat #%s', $post_id )
			]
		);
		update_post_meta( $post_id, '_magicai_messages', [
			[
				"role" => "system",
				"content" => "You are a helpful assistant."
			]
		] );
		wp_send_json_success( $post_id );
	}

}

/**
 * Save chat data for a chat post.
 *
 * This function saves user and system messages in a chat post's metadata.
 *
 * @since 1.0.0
 */
function magicai_save_chat_data() {

	$post_id = intval( $_POST['post_id'] );
	$prompt = $_POST['prompt'];
	$message = $_POST['message'];
	$pdf = intval( $_POST['pdf'] );

	$data = get_post_meta( $post_id, '_magicai_messages', true ) ?? array();

	if ( ! empty( $post_id ) ) {
		$data[] = [ 'role' => 'user', 'content'=> $prompt ];
		$data[] = [ 'role' => 'system', 'content'=> $message ];
		update_post_meta( $post_id, '_magicai_messages', $data );
		
		if ( $pdf ) {
			$pdf_files = get_post_meta( $post_id, '_magicai_pdf_file', true );
			if ( ! is_array( $pdf_files ) ) {
				$pdf_files = array();
			}
			$pdf_files[] = array(
				'attachment_id' => $pdf,
				'attachment_url' => wp_get_attachment_url( $pdf ),
				'attachment_filename' => basename( get_attached_file( $pdf ) ),
				'message' => $prompt,
			);
			update_post_meta( $post_id, '_magicai_pdf_file', $pdf_files );
		}
	}

	wp_send_json_success( $data );

}

/**
 * Get chat data for a chat post.
 *
 * This function retrieves chat messages from a chat post's metadata.
 *
 * @since 1.0.0
 */
function magicai_get_chat_data() {

	$post_id = intval( $_POST['post_id'] );
	$data = get_post_meta( $post_id, '_magicai_messages', true );
	wp_send_json_success( $data );

}

/**
 * Get chat messages for a chat post and generate HTML for display.
 *
 * This function retrieves chat messages from a chat post's metadata and generates HTML for display.
 *
 * @since 1.0.0
 */
function magicai_get_chat() {

	$post_id = intval( $_POST['post_id'] );
	$data = get_post_meta( $post_id, '_magicai_messages', true );
	$pdf_file = get_post_meta( $post_id, '_magicai_pdf_file', true );
	$html = '';

	foreach ( $data as $message ) {
		if( $message['role'] === 'user' ) {
			if ( is_array( $message['content'] ) ) {
				foreach( $message['content'] as $msg ) {
					if ( isset( $msg['text'] ) ) {
						$msg['text'] = preg_replace('/(?:\r\n|\r|\n)/', ' <br> ', $msg['text']);
						$html .= sprintf(
							'<div class="magicai-chat--message">
								<div class="text">
									<img width="32" src="%s">
									%s
								</div>
							</div>',
							esc_url( get_avatar_url( get_current_user_id() ) ),
							$msg['text']
						);
					} elseif( isset( $msg['image_url']['url'] ) ) {
						$html .= sprintf(
							'<div class="magicai-chat--message image">
								<div class="text">
									<img class="ai" src="%s"/>
								</div>
							</div>',
							$msg['image_url']['url']
						);
					}
				}
			} else {
				$message['content'] = preg_replace('/(?:\r\n|\r|\n)/', ' <br> ', $message['content']);
				$html .= sprintf(
					'<div class="magicai-chat--message">
						<div class="text">
							<img width="32" src="%s">
							%s
						</div>
					</div>',
					esc_url( get_avatar_url( get_current_user_id() ) ),
					$message['content']
				);
				$message_index = array_search($message['content'], array_column($pdf_file, 'message'));
				if ($message_index !== false) {
					$html .= sprintf(
						'<div class="magicai-chat--message image">
							<a class="text pdf" href="%1$s" target="_blank">
								<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" fill="none"><path fill="#E9E9E0" d="M23.776 0H5.12c-.52 0-.94.421-.94 1.238v34.12c0 .22.42.642.94.642h25.762c.52 0 .94-.421.94-.643V8.343c0-.447-.06-.591-.165-.697l-7.48-7.48a.568.568 0 0 0-.4-.166Z"/><path fill="#D9D7CA" d="M24.107.097v7.617h7.618L24.107.097Z"/><path fill="#CC4B4C" d="M12.544 21.423c-.223 0-.438-.073-.621-.21-.67-.502-.76-1.06-.717-1.441.117-1.047 1.411-2.142 3.848-3.258.966-2.12 1.886-4.73 2.435-6.91-.642-1.397-1.265-3.209-.81-4.271.159-.373.357-.658.728-.782.147-.048.517-.11.653-.11.324 0 .609.417.81.674.19.242.62.754-.239 4.373.867 1.79 2.095 3.613 3.271 4.861.843-.152 1.568-.23 2.159-.23 1.006 0 1.616.235 1.865.718.206.4.122.867-.25 1.389-.358.5-.852.765-1.428.765-.781 0-1.692-.493-2.707-1.468a30.8 30.8 0 0 0-5.675 1.814c-.537 1.14-1.052 2.059-1.532 2.732-.659.923-1.227 1.354-1.79 1.354Zm1.712-3.296c-1.374.772-1.934 1.407-1.974 1.764-.007.06-.024.215.277.445.096-.03.655-.285 1.697-2.209Zm8.767-2.855c.523.403.651.607.994.607.15 0 .58-.007.778-.284.096-.134.133-.22.148-.267-.08-.041-.184-.126-.756-.126-.324 0-.733.014-1.165.07ZM18.22 11.04a45.826 45.826 0 0 1-1.719 4.863 32.123 32.123 0 0 1 4.176-1.299c-.867-1.007-1.735-2.266-2.457-3.563Zm-.39-5.44c-.063.022-.855 1.13.062 2.068.61-1.36-.034-2.076-.062-2.067ZM30.881 36H5.12a.94.94 0 0 1-.94-.94V25.07h27.643v9.988a.94.94 0 0 1-.94.941Z"/><path fill="#fff" d="M11.176 34.071h-1.055v-6.477h1.863c.275 0 .548.044.817.132.27.088.511.22.725.395.214.176.387.388.52.637.13.249.197.529.197.84 0 .328-.056.625-.167.892a1.866 1.866 0 0 1-.466.673c-.2.18-.44.322-.72.421-.282.1-.593.15-.932.15h-.783l.001 2.337Zm0-5.677v2.566h.967c.128 0 .256-.022.382-.066a.962.962 0 0 0 .348-.216c.105-.1.19-.238.254-.417a1.976 1.976 0 0 0 .053-1.028c-.03-.137-.09-.27-.18-.395a1.066 1.066 0 0 0-.383-.316c-.164-.085-.38-.128-.65-.128h-.791ZM20.712 30.653c0 .533-.057.988-.172 1.366a3.395 3.395 0 0 1-.435.95 2.236 2.236 0 0 1-.593.602c-.22.147-.432.256-.637.33-.205.073-.393.12-.563.14-.17.02-.295.03-.378.03h-2.452v-6.477h1.951c.546 0 1.025.087 1.437.26.413.171.756.402 1.029.689.272.287.476.614.61.98.136.366.203.743.203 1.13Zm-3.129 2.645c.715 0 1.23-.228 1.547-.685.316-.457.474-1.12.474-1.987 0-.269-.032-.536-.096-.8a1.711 1.711 0 0 0-.373-.716 1.971 1.971 0 0 0-.752-.518c-.316-.132-.726-.198-1.23-.198h-.616v4.904h1.046ZM23.314 28.394v2.039h2.707v.72h-2.707v2.918H22.24v-6.477h4.052v.8h-2.98Z"/></svg>
								%2$s
							</a>
						</div>',
						$pdf_file[$message_index]['attachment_url'],
						$pdf_file[$message_index]['attachment_filename'],
					);
				}
			}
		} else {
			$message['content'] = preg_replace('/(?:\r\n|\r|\n)/', ' <br> ', $message['content']);
			if ( get_post_type( $post_id ) == 'magicai-vision' ) {
				$first_message = 'Seamlessly upload any image you want to explore or discuss and get insightful conversations.';
			} elseif ( get_post_type( $post_id ) == 'magicai-chat-pdf' ) {
				$first_message = 'Seamlessly upload any PDF you want to explore or discuss and get insightful conversations.';
			} else {
				$first_message = 'How can I Help you today?';
			}
			$html .= sprintf(
				'<div class="magicai-chat--message ai">
					<div class="text">
						<img width="32" src="%s">
						<div>%s</div>
					</div>
					<div class="magicai-chat--message-action">
						<div class="btn copy">
							<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M6.9998 6V3C6.9998 2.44772 7.44752 2 7.9998 2H19.9998C20.5521 2 20.9998 2.44772 20.9998 3V17C20.9998 17.5523 20.5521 18 19.9998 18H16.9998V20.9991C16.9998 21.5519 16.5499 22 15.993 22H4.00666C3.45059 22 3 21.5554 3 20.9991L3.0026 7.00087C3.0027 6.44811 3.45264 6 4.00942 6H6.9998ZM5.00242 8L5.00019 20H14.9998V8H5.00242ZM8.9998 6H16.9998V16H18.9998V4H8.9998V6ZM7 11H13V13H7V11ZM7 15H13V17H7V15Z"></path></svg>
							<span>Copy</span>
						</div>
					</div>
				</div>',
				esc_url( MAGICAI_URL . 'assets/img/logo.svg' ),
				str_replace( 'You are a helpful assistant.', $first_message, $message['content'] )
			);
		}

	}
	wp_send_json_success( $html );

}

/**
 * Delete a chat post.
 *
 * This function deletes a chat post identified by the provided post ID and logs the deletion action.
 *
 * @since 1.0.0
 */
function magicai_delete_chat() {

	$post_id = intval( $_POST['post_id'] );
	MagicAI_Logs::instance()->add_log(
		'chat_actions',
		'Chat Deleted',
		get_the_title( $post_id ),
	);
	wp_delete_post( $post_id );
	wp_send_json_success( 'Chat Deleted' );

}

/**
 * Edit the name of a chat post.
 *
 * This function updates the title of a chat post identified by the provided post ID and logs the name update action.
 *
 * @since 1.0.0
 */
function magicai_edit_chat_name() {

	$post_id = intval( $_POST['post_id'] );
	$title = sanitize_text_field( $_POST['title'] );
	MagicAI_Logs::instance()->add_log(
		'chat_actions',
		'Chat Name Updated',
		sprintf( '%s -> %s', get_the_title( $post_id ), $title ),
	);
	wp_update_post( [ 
		'ID' => $post_id,
		'post_title' => $title 
	] );
	wp_send_json_success( 'Chat Name Updated' );

}

/**
 * Allow additional audio MIME types.
 *
 * This function adds custom audio MIME types to the list of allowed MIME types.
 *
 * @param array $mimes An array of allowed MIME types.
 * @return array The modified array of allowed MIME types.
 *
 * @since 1.0.0
 */
function magicai_allow_audio_mimes( $mimes ) {
	foreach( magicai_helper()->custom_mime_types() as $type => $ext ) {
		$mimes[$type] = $ext;
	}
    return $mimes;
}

/**
 * Fix the result of wp_check_filetype for audio MIME types.
 *
 * This function corrects the result of `wp_check_filetype` for audio MIME types, ensuring that the 'ext' and 'type' are properly set.
 *
 * @param array $data The data array containing file information.
 * @param string $file The file path.
 * @param string $filename The file name.
 * @param array $mimes An array of MIME types.
 * @return array The modified data array with corrected 'ext' and 'type'.
 *
 * @since 1.0.0
 */
function magicai_fix_wp_check_filetype_and_ext( $data, $file, $filename, $mimes ) {
	if ( ! empty( $data['ext'] ) && ! empty( $data['type'] ) ) {
		return $data;
	}

	$types = magicai_helper()->custom_mime_types();
	$filetype = wp_check_filetype( $filename, $mimes );

	if ( ! isset( $types[ $filetype['ext'] ] ) ) {
		return $data;
	}
	$filetype['type'] = explode( '|', $filetype['type'] )[0];

	return [
		'ext' => $filetype['ext'],
		'type' => $filetype['type'],
		'proper_filename' => $data['proper_filename'],
	];
}

/**
 * Modify the row actions for a specific post type.
 *
 * @param array   $actions An array of row action links.
 * @param WP_Post $post    The current post object.
 * @return array Modified array of row action links.
 * @since 1.0.0
 */
add_filter( 'post_row_actions', function( $actions, $post ) {
	if ( $post->post_type == "magicai-documents" ) {

		// TODO: Improve Edit option for code, voiceover and speech-to-text
		$actions = array(
			'edit' => $actions['edit'],
			'trash' => $actions['trash'],
		);

	}

	return $actions;
}, 10, 2 );

/**
 * Modify the columns displayed in the admin panel for 'magicai-documents' post type.
 *
 * @param array $columns An array of column names.
 * @return array Modified array of column names.
 * @since 1.0.0
 */
add_filter('manage_magicai-documents_posts_columns', function($columns) {

	unset($columns['title']);
	unset($columns['date']);
	$columns['new_title'] = 'Title';
	$columns['type'] = 'Type';
	$columns['content'] = 'Content';
	$columns['author'] = 'Author';
	$columns['date'] = 'Date';

	return $columns;
});

/**
 * Customize the content of custom columns in the admin panel for 'magicai-documents' post type.
 *
 * @param string $column_key The identifier for the custom column.
 * @param int    $post_id    The ID of the current post.
 * @since 1.0.0
 */
add_action('manage_magicai-documents_posts_custom_column', function($column_key, $post_id) {

	switch ( $column_key ) {
		case 'type' :
			$type = get_post_meta($post_id, '_magicai_doc_type', true);
	
			if ( $type == 'transcribe' ) {
				?>
					<span class="transcribe">
						<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M11.9998 3C10.3429 3 8.99976 4.34315 8.99976 6V10C8.99976 11.6569 10.3429 13 11.9998 13C13.6566 13 14.9998 11.6569 14.9998 10V6C14.9998 4.34315 13.6566 3 11.9998 3ZM11.9998 1C14.7612 1 16.9998 3.23858 16.9998 6V10C16.9998 12.7614 14.7612 15 11.9998 15C9.23833 15 6.99976 12.7614 6.99976 10V6C6.99976 3.23858 9.23833 1 11.9998 1ZM3.05469 11H5.07065C5.55588 14.3923 8.47329 17 11.9998 17C15.5262 17 18.4436 14.3923 18.9289 11H20.9448C20.4837 15.1716 17.1714 18.4839 12.9998 18.9451V23H10.9998V18.9451C6.82814 18.4839 3.51584 15.1716 3.05469 11Z"></path></svg>
						<?php esc_html_e( 'Speech to Text', 'magicai-wp' ); ?>
					</span>
				<?php
			} elseif ( $type == 'voiceover' ) {
				?>
					<span class="voiceover">
						<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M16.9337 8.96494C16.426 5.03562 13.0675 2 9 2 4.58172 2 1 5.58172 1 10 1 11.8924 1.65707 13.6313 2.7555 15.0011 3.56351 16.0087 4.00033 17.1252 4.00025 18.3061L4 22H13L13.001 19H15C16.1046 19 17 18.1046 17 17V14.071L18.9593 13.2317C19.3025 13.0847 19.3324 12.7367 19.1842 12.5037L16.9337 8.96494ZM3 10C3 6.68629 5.68629 4 9 4 12.0243 4 14.5665 6.25141 14.9501 9.22118L15.0072 9.66262 16.5497 12.0881 15 12.7519V17H11.0017L11.0007 20H6.00013L6.00025 18.3063C6.00036 16.6672 5.40965 15.114 4.31578 13.7499 3.46818 12.6929 3 11.3849 3 10ZM21.1535 18.1024 19.4893 16.9929C20.4436 15.5642 21 13.8471 21 12.0001 21 10.153 20.4436 8.4359 19.4893 7.00722L21.1535 5.89771C22.32 7.64386 23 9.74254 23 12.0001 23 14.2576 22.32 16.3562 21.1535 18.1024Z"></path></svg>
						<?php esc_html_e( 'Voiceover', 'magicai-wp' ); ?>
					</span>
				<?php
			} elseif ( $type == 'code' ) {
				?>
					<span class="code">
						<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M24 12L18.3431 17.6569L16.9289 16.2426L21.1716 12L16.9289 7.75736L18.3431 6.34315L24 12ZM2.82843 12L7.07107 16.2426L5.65685 17.6569L0 12L5.65685 6.34315L7.07107 7.75736L2.82843 12ZM9.78845 21H7.66009L14.2116 3H16.3399L9.78845 21Z"></path></svg>
						<?php esc_html_e( 'Code', 'magicai-wp' ); ?>
					</span>
				<?php
			} elseif ( $type == 'post' ) {
				?>
					<span class="post">
						<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M20 22H4C3.44772 22 3 21.5523 3 21V3C3 2.44772 3.44772 2 4 2H20C20.5523 2 21 2.44772 21 3V21C21 21.5523 20.5523 22 20 22ZM19 20V4H5V20H19ZM7 6H11V10H7V6ZM7 12H17V14H7V12ZM7 16H17V18H7V16ZM13 7H17V9H13V7Z"></path></svg>
						<?php esc_html_e( 'Post', 'magicai-wp' ); ?>
					</span>
				<?php
			} elseif ( $type == 'product' ) {
				?>
					<span class="product">
						<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M4.00436 6.41662L0.761719 3.17398L2.17593 1.75977L5.41857 5.00241H20.6603C21.2126 5.00241 21.6603 5.45012 21.6603 6.00241C21.6603 6.09973 21.6461 6.19653 21.6182 6.28975L19.2182 14.2898C19.0913 14.7127 18.7019 15.0024 18.2603 15.0024H6.00436V17.0024H17.0044V19.0024H5.00436C4.45207 19.0024 4.00436 18.5547 4.00436 18.0024V6.41662ZM6.00436 7.00241V13.0024H17.5163L19.3163 7.00241H6.00436ZM5.50436 23.0024C4.67593 23.0024 4.00436 22.3308 4.00436 21.5024C4.00436 20.674 4.67593 20.0024 5.50436 20.0024C6.33279 20.0024 7.00436 20.674 7.00436 21.5024C7.00436 22.3308 6.33279 23.0024 5.50436 23.0024ZM17.5044 23.0024C16.6759 23.0024 16.0044 22.3308 16.0044 21.5024C16.0044 20.674 16.6759 20.0024 17.5044 20.0024C18.3328 20.0024 19.0044 20.674 19.0044 21.5024C19.0044 22.3308 18.3328 23.0024 17.5044 23.0024Z"></path></svg>
						<?php esc_html_e( 'Product', 'magicai-wp' ); ?>
					</span>
				<?php
			} else {
				?>
					<span class="text">
						<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M20 22H4C3.44772 22 3 21.5523 3 21V3C3 2.44772 3.44772 2 4 2H20C20.5523 2 21 2.44772 21 3V21C21 21.5523 20.5523 22 20 22ZM19 20V4H5V20H19ZM7 6H11V10H7V6ZM7 12H17V14H7V12ZM7 16H17V18H7V16ZM13 7H17V9H13V7Z"></path></svg>
						<?php esc_html_e( 'Text', 'magicai-wp' ); ?>
					</span>
				<?php
			}
		break;
		case 'content':
			echo html_entity_decode( wp_trim_words( htmlentities( get_the_content() ), 20, '...' ) );
			if ( get_post_meta($post_id, '_magicai_doc_type', true)  == 'voiceover' ) {
				printf( 
					'<div class="data-audio" data-audio="%s"><div class="audio-preview"></div></div>',
					wp_get_attachment_url( get_post_meta( $post_id, '_magicai_attachment_id', true ) )
				);
			}
		break;
		case 'new_title':
			echo get_the_title();
		break;
	}
}, 10, 2);

/**
 * Make the 'type' column sortable for 'magicai-documents' post type.
 *
 * @param array $columns An array of sortable columns.
 * @return array Modified array of sortable columns.
 * @since 1.0.0
 */
add_filter('manage_edit-magicai-documents_sortable_columns', function($columns) {
	$columns['type'] = 'type';
	return $columns;
});

/**
 * Modify the query for 'magicai-documents' post type based on 'type' column sorting.
 *
 * @param WP_Query $query The WP_Query instance.
 * @since 1.0.0
 */
add_action('pre_get_posts', function($query) {
	if (!is_admin()) {
		return;
	}
 
	$orderby = $query->get('orderby');
	if ($orderby == 'type') {
		$query->set('meta_key', '_magicai_doc_type');
		$query->set('orderby', 'meta_value');
	}
});

/**
 * Add custom dropdown filter for 'magicai-documents' post type.
 * 
 * @since 1.0.0
 */
add_action('restrict_manage_posts', function () {
    global $typenow;
	
    if ($typenow === 'magicai-documents') {
        $selected = isset($_GET['_magicai_doc_type']) ? $_GET['_magicai_doc_type'] : '';
        $options = array(
            'transcribe' => esc_html__('Speech to Text', 'magicai-wp'),
            'voiceover' => esc_html__('Voiceover', 'magicai-wp'),
            'code' => esc_html__('Code', 'magicai-wp'),
            'text' => esc_html__('Text', 'magicai-wp'),
            'post' => esc_html__('Post', 'magicai-wp'),
            'product' => esc_html__('Product', 'magicai-wp'),
        );
        echo '<select name="_magicai_doc_type">';
        echo '<option value="">' . esc_html__('All Types', 'magicai-wp') . '</option>';
        
        foreach ($options as $value => $label) {
            echo '<option value="' . esc_attr($value) . '" ' . selected($selected, $value, false) . '>' . $label . '</option>';
        }
        
        echo '</select>';
    }
});

/**
 * Modify the query for 'magicai-documents' post type based on custom filter.
 *
 * @param WP_Query $query The WP_Query instance.
 * @since 1.0.0
 */
add_action('pre_get_posts', function ($query) {
    global $pagenow;
    global $typenow;

    if ($pagenow == 'edit.php' && $typenow == 'magicai-documents' && isset($_GET['_magicai_doc_type']) && $_GET['_magicai_doc_type'] != '') {
        $query->query_vars['meta_key'] = '_magicai_doc_type';
        $query->query_vars['meta_value'] = sanitize_text_field($_GET['_magicai_doc_type']);
    }
});


/**
 * Display a notice in the admin area for the 'magicai-chatbot' screen.
 *
 * This function adds an admin notice and displays it only on the 'magicai-chatbot' screen.
 * @since 1.0.0
 */
add_action('admin_notices', function () {
    $screen = get_current_screen();

	if( $screen->id !='magicai-chatbot' ){
		return;
	} else {
	?>
		<div class="wrap">
			<h1 class="wp-heading-inline show" style="display:inline-block;"><?php esc_html_e( 'ChatBot', 'magicai-wp' ); ?></h1>
			<a href="<?php echo admin_url('admin.php?page=magicai-chatbot'); ?>" class="page-title-action show"><?php esc_html_e( 'Go Back to ChatBot', 'magicai-wp' ); ?></a>
		</div>

		<style scoped>
		.wp-heading-inline:not(.show),
		.page-title-action:not(.show) { display:none !important;}
		</style>
	<?php
	}
} );

/**
 * AJAX callback for saving MagicAI chatbot settings.
 *
 * This function handles AJAX requests for updating MagicAI chatbot settings.
 * It checks the AJAX referer for security, then updates the 'magicai_chatbot_settings' option
 * with sanitized values obtained from the POST request. Finally, it sends a JSON response.
 *
 * @return void Sends JSON response indicating success or failure of settings update.
 * 
 * @since 1.0.0
 * 
 */
add_action('wp_ajax_magicai_chatbot_settings', function(){
	if ( check_ajax_referer( 'magicai_chatbot_settings', 'nonce', false ) ){

		update_option( 'magicai_chatbot_settings', [
			'status'            => sanitize_text_field( $_POST['status'] ),
			'template'          => sanitize_text_field( $_POST['template'] ),
			'position'          => sanitize_text_field( $_POST['position'] ),
			'limit'             => sanitize_text_field( $_POST['limit'] ),
			'limit_per_seconds' => sanitize_text_field( $_POST['limit_per_seconds'] ),
			'is_user_logged_in' => sanitize_text_field( $_POST['is_user_logged_in'] ),
		]);

		MagicAI_Logs::instance()->add_log(
			'chatbot',
			'Settings',
			'Settings Saved',
		);

		wp_send_json( [ 
			'message' => esc_html__( 'Settings Saved Successfully! Redirecting...', 'magicai-wp' ),
		] );
	}

	wp_send_json( [ 
		'error' => true ,
		'message' => esc_html__( 'Security Failed!', 'magicai-wp' )
	] );
});

/**
 * Start a new chat with the MagicAI ChatBot.
 *
 * This function initiates a new chat session with the MagicAI ChatBot. It retrieves the necessary settings,
 * creates a new chat post, and sends back the chat ID along with the initial chat messages to the client.
 *
 * @since 1.3
 */
function magicai_chatbot_start_new_chat() {

	if ( magicai_helper()->check_user_ip_status() ) {
		wp_send_json( [
			'error' => true,
			'message' => esc_html__('You have been banned by the administrator. Contact the administrator for details.', 'magicai-wp'),
		] );
	}
  
	$html = '';

	$chatbot_options = get_option('magicai_chatbot_settings', array() );
	
	if ( ! $chatbot_options['template'] ) {
		wp_send_json( [
			'error' => true,
			'message' => esc_html__('Select a ChatBot Template!', 'magicai-wp'),
		] );
	}
			
	$instructions = get_post_meta( $chatbot_options['template'], '_instructions', true );	
	$first_message = get_post_meta( $chatbot_options['template'], '_first_message', true );	

	// Create New Chat
	$post_id = wp_insert_post(
		[
			'post_type' => 'magicai-chat',
			'post_status' => 'publish',
			'meta_input' => [ 
				'_magicai_chat_type' => 'chatbot',
				'_magicai_chat_first_message' => $first_message,
				'_magicai_chatbot_template_id' => $chatbot_options['template'],
				'_magicai_user_ip' => magicai_helper()->get_ip() ?? $_SERVER['REMOTE_ADDR'],
				'_magicai_user_id' => get_current_user_id(),
				'_magicai_messages' => [
					[
						"role" => "system",
						"content" => $instructions,
					]
				]
			]
		]
	);

	if( ! is_wp_error( $post_id ) ) {
		wp_update_post(
			[
				'ID' => $post_id,
				'post_title'=> sprintf( 'ChatBot #%s', $post_id )
			]
		);
	}

	$chat_id = $post_id;

	MagicAI_Logs::instance()->add_log(
		'chatbot',
		'Chat Started',
		'ChatBot #'.$chat_id,
	);

	$data = get_post_meta( $chat_id, '_magicai_messages', true );

	$first_message = true;
	foreach ( $data as $message ) {

		if( $message['role'] === 'user' ) {
			$message['content'] = preg_replace('/(?:\r\n|\r|\n)/', ' <br> ', $message['content']);
			$html .= sprintf(
				'<div class="magicai-chatbot-widget--message">%s</div>',
				$message['content']
			);
		} else {
			$message['content'] = preg_replace('/(?:\r\n|\r|\n)/', ' <br> ', $message['content']);
			$html .= sprintf(
				'<div class="magicai-chatbot-widget--message ai">%s</div>',
				$first_message ? get_post_meta( $chat_id, '_magicai_chat_first_message', true ) : $message['content']
			);

			if ( $first_message ) {
				$first_message = false;
			}
		}
		
	}

	$bot_image = get_post_meta( get_option('magicai_chatbot_settings', array() )['template'], '_image', true );
	$bot_name = get_post_meta( get_option('magicai_chatbot_settings', array() )['template'], '_name', true );
	$bot_role = get_post_meta( get_option('magicai_chatbot_settings', array() )['template'], '_role', true );
	// $data = get_post_meta( $chat_id, '_magicai_messages', true );
	$html2 = sprintf(
		'<div class="magicai-chatbot-widget--message-list--item" data-id="%5$s">
			<img src="%1$s" class="%2$s" alt="ChatBot">
			<div class="magicai-chatbot-widget--message-list--item-title">
				%3$s<br>
				<span>%4$s</span>
			</div>
			<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M13.1717 12.0007L8.22192 7.05093L9.63614 5.63672L16.0001 12.0007L9.63614 18.3646L8.22192 16.9504L13.1717 12.0007Z"></path></svg>
		</div>',
		esc_attr( $bot_image ),
		esc_attr( strpos( basename( $bot_image ), '.svg' ) !== false ? 'svg' : '' ),
		// esc_html( wp_trim_words( end($data)['content'], 8 ) ),
		esc_html__( 'New Chat', 'magicai-wp' ),
		esc_html( $bot_name . ' &bull; ' . $bot_role ),
		esc_attr( $chat_id )

	);

	wp_send_json( [
		'chat_id' => $chat_id,
		'output' => $html,
		'output_list' => $html2
	] );

}

/**
 * Retrieves or initializes a chat session for the user or IP address.
 *
 * Determines the user ID if logged in; otherwise, retrieves the IP address.
 * Checks if a chat ID exists; if not, creates a new chat session based on
 * the selected ChatBot template and initializes it with system instructions.
 * Retrieves chat messages for the chat session and prepares HTML output
 * for the chat widget, differentiating user and system messages.
 * Sends a JSON response containing HTML output for the chat widget.
 *
 * @return void Sends a JSON response with HTML output for the chat widget.
 * 
 * @since 1.0.0
 */
function magicai_chatbot_get_chat() {
  
	if ( is_user_logged_in() ) {
		$user_or_ip = get_current_user_id();
	} else {
		$user_or_ip = magicai_helper()->get_ip() ?? $_SERVER['REMOTE_ADDR'];
	}

	$chat_id = !empty( $_POST['chat_id'] ) ? intval( $_POST['chat_id'] ) : '';

	$chat_history = false;
	$html = '';

	$chats = get_posts( [
		'post_type' => "magicai-chat",
		'posts_per_page' => -1,
		'meta_query' => [
			[
				'key' => '_magicai_user_ip',
				'value' => magicai_helper()->get_ip() ?? $_SERVER['REMOTE_ADDR'],
				'compare' => '=',
			]
		]
	] );

	if ( empty( $chat_id ) ) {
		// Check Chat if exists
		if ( !$chats ) {
			$chatbot_options = get_option('magicai_chatbot_settings', array() );
	
			if ( ! $chatbot_options['template'] ) {
				wp_send_json( [
					'error' => true,
					'message' => esc_html__('Select a ChatBot Template!', 'magicai-wp'),
				] );
			}
			
			$instructions = get_post_meta( $chatbot_options['template'], '_instructions', true );	
			$first_message = get_post_meta( $chatbot_options['template'], '_first_message', true );	
	
			// Create New Chat
			$post_id = wp_insert_post(
				[
					'post_type' => 'magicai-chat',
					'post_status' => 'publish',
					'meta_input' => [ 
						'_magicai_chat_type' => 'chatbot',
						'_magicai_chat_first_message' => $first_message,
						'_magicai_chatbot_template_id' => $chatbot_options['template'],
						'_magicai_user_ip' => magicai_helper()->get_ip() ?? $_SERVER['REMOTE_ADDR'],
						'_magicai_user_id' => get_current_user_id(),
						'_magicai_messages' => [
							[
								"role" => "system",
								"content" => $instructions,
							]
						]
					]
				]
			);
	
			if( ! is_wp_error( $post_id ) ) {
				wp_update_post(
					[
						'ID' => $post_id,
						'post_title'=> sprintf( 'ChatBot #%s', $post_id )
					]
				);
			}
	
			$chat_id = $post_id;
	
			MagicAI_Logs::instance()->add_log(
				'chatbot',
				'Chat Started',
				'ChatBot #'.$chat_id,
			);
		} else {
			$chat_history = true;
	
			$bot_image = get_post_meta( get_option('magicai_chatbot_settings', array() )['template'], '_image', true );
			$bot_name = get_post_meta( get_option('magicai_chatbot_settings', array() )['template'], '_name', true );
			$bot_role = get_post_meta( get_option('magicai_chatbot_settings', array() )['template'], '_role', true );
			foreach ( $chats as $chat ) {
				$data = get_post_meta( $chat->ID, '_magicai_messages', true );
				$html .= sprintf(
					'<div class="magicai-chatbot-widget--message-list--item" data-id="%5$s">
						<img src="%1$s" class="%2$s" alt="ChatBot">
						<div class="magicai-chatbot-widget--message-list--item-title">
							%3$s<br>
							<span>%4$s</span>
						</div>
						<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M13.1717 12.0007L8.22192 7.05093L9.63614 5.63672L16.0001 12.0007L9.63614 18.3646L8.22192 16.9504L13.1717 12.0007Z"></path></svg>
					</div>',
					esc_attr( $bot_image ),
					esc_attr( strpos( basename( $bot_image ), '.svg' ) !== false ? 'svg' : '' ),
					esc_html( wp_trim_words( end($data)['content'], 7 ) ),
					esc_html( $bot_name . ' &bull; ' . $bot_role ),
					esc_attr( $chat->ID )
	
				);
			}
	
			wp_send_json( [
				'chat_history' => true,
				'output' => $html
			] );
		}
	}

	$data = get_post_meta( $chat_id, '_magicai_messages', true );

	$first_message = true;
	foreach ( $data as $message ) {

		if( $message['role'] === 'user' ) {
			$message['content'] = preg_replace('/(?:\r\n|\r|\n)/', ' <br> ', $message['content']);
			$html .= sprintf(
				'<div class="magicai-chatbot-widget--message">%s</div>',
				$message['content']
			);
		} else {
			$message['content'] = preg_replace('/(?:\r\n|\r|\n)/', ' <br> ', $message['content']);
			$html .= sprintf(
				'<div class="magicai-chatbot-widget--message ai">%s</div>',
				$first_message ? get_post_meta( $chat_id, '_magicai_chat_first_message', true ) : $message['content']
			);

			if ( $first_message ) {
				$first_message = false;
			}
		}
		
	}

	wp_send_json( [
		'chat_id' => $chat_id,
		'output' => $html
	] );

}

/**
 * Retrieves or initializes a chat session for the user or IP address.
 *
 * Determines the user ID if logged in; otherwise, retrieves the IP address.
 * Checks if a chat ID exists; if not, creates a new chat session based on
 * the selected ChatBot template and initializes it with system instructions.
 * Retrieves chat messages for the chat session and prepares HTML output
 * for the chat widget, differentiating user and system messages.
 * Sends a JSON response containing HTML output for the chat widget.
 *
 * @return void Sends a JSON response with HTML output for the chat widget.
 * 
 * @since 1.4
 */
function magicai_chatbot_get_chat_conversations() {
  
	if ( is_user_logged_in() ) {
		$user_or_ip = get_current_user_id();
	} else {
		$user_or_ip = magicai_helper()->get_ip() ?? $_SERVER['REMOTE_ADDR'];
	}

	$chat_id = !empty( $_POST['chat_id'] ) ? intval( $_POST['chat_id'] ) : '';

	$chat_history = false;
	$html = '';

	$chats = get_posts( [
		'post_type' => "magicai-chat",
		'posts_per_page' => -1,
		'meta_query' => [
			[
				'key' => '_magicai_user_ip',
				'value' => magicai_helper()->get_ip() ?? $_SERVER['REMOTE_ADDR'],
				'compare' => '=',
			]
		]
	] );

	if ( empty( $chat_id ) ) {
		// Check Chat if exists
		if ( !$chats ) {
			$chatbot_options = get_option('magicai_chatbot_settings', array() );
	
			if ( ! $chatbot_options['template'] ) {
				wp_send_json( [
					'error' => true,
					'message' => esc_html__('Select a ChatBot Template!', 'magicai-wp'),
				] );
			}
			
			$instructions = get_post_meta( $chatbot_options['template'], '_instructions', true );	
			$first_message = get_post_meta( $chatbot_options['template'], '_first_message', true );	
	
			// Create New Chat
			$post_id = wp_insert_post(
				[
					'post_type' => 'magicai-chat',
					'post_status' => 'publish',
					'meta_input' => [ 
						'_magicai_chat_type' => 'chatbot',
						'_magicai_chat_first_message' => $first_message,
						'_magicai_chatbot_template_id' => $chatbot_options['template'],
						'_magicai_user_ip' => magicai_helper()->get_ip() ?? $_SERVER['REMOTE_ADDR'],
						'_magicai_user_id' => get_current_user_id(),
						'_magicai_messages' => [
							[
								"role" => "system",
								"content" => $instructions,
							]
						]
					]
				]
			);
	
			if( ! is_wp_error( $post_id ) ) {
				wp_update_post(
					[
						'ID' => $post_id,
						'post_title'=> sprintf( 'ChatBot #%s', $post_id )
					]
				);
			}
	
			$chat_id = $post_id;
	
			MagicAI_Logs::instance()->add_log(
				'chatbot',
				'Chat Started',
				'ChatBot #'.$chat_id,
			);
		} else {
			$chat_history = true;
	
			$bot_image = get_post_meta( get_option('magicai_chatbot_settings', array() )['template'], '_image', true );
			$bot_name = get_post_meta( get_option('magicai_chatbot_settings', array() )['template'], '_name', true );
			$bot_role = get_post_meta( get_option('magicai_chatbot_settings', array() )['template'], '_role', true );
			foreach ( $chats as $chat ) {
				$data = get_post_meta( $chat->ID, '_magicai_messages', true );
				$html .= sprintf(
					'<div class="magicai-chatbot-widget--message-list--item" data-id="%5$s">
						<img src="%1$s" class="%2$s" alt="ChatBot">
						<div class="magicai-chatbot-widget--message-list--item-title">
							%3$s<br>
							<span>%4$s</span>
						</div>
						<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M13.1717 12.0007L8.22192 7.05093L9.63614 5.63672L16.0001 12.0007L9.63614 18.3646L8.22192 16.9504L13.1717 12.0007Z"></path></svg>
					</div>',
					esc_attr( $bot_image ),
					esc_attr( strpos( basename( $bot_image ), '.svg' ) !== false ? 'svg' : '' ),
					esc_html( wp_trim_words( end($data)['content'], 7 ) ),
					esc_html( $bot_name . ' &bull; ' . $bot_role ),
					esc_attr( $chat->ID )
	
				);
			}
	
			wp_send_json( [
				'chat_history' => true,
				'output' => $html
			] );
		}
	}

	$data = get_post_meta( $chat_id, '_magicai_messages', true );

	$first_message = true;
	foreach ( $data as $message ) {

		if( $message['role'] === 'user' ) {
			$message['content'] = preg_replace('/(?:\r\n|\r|\n)/', ' <br> ', $message['content']);
			$html .= sprintf(
				'<div class="magicai-chatbot-widget--message">%s</div>',
				$message['content']
			);
		} else {
			$message['content'] = preg_replace('/(?:\r\n|\r|\n)/', ' <br> ', $message['content']);
			$html .= sprintf(
				'<div class="magicai-chatbot-widget--message ai">%s</div>',
				$first_message ? get_post_meta( $chat_id, '_magicai_chat_first_message', true ) : $message['content']
			);

			if ( $first_message ) {
				$first_message = false;
			}
		}
		
	}

	$user_id = get_post_meta( $chat_id, '_magicai_user_id', true );
	$user_data = get_userdata( $user_id );
	$user_name = $user_data->display_name ? $user_data->display_name : '';
	$user_ip = get_post_meta( $chat_id, '_magicai_user_ip', true );

	$html_header = sprintf(
		'<div class="details">
			<span class="title">%1$s</span>
			<span class="detail">%2$s: %3$s</span>
			<span class="detail">%4$s: %5$s</span>
		</div>
		<div class="details">
			<span class="title">%6$s</span>
			<span class="detail">%7$s: %8$s</span>
			<span class="detail">%9$s: %10$s</span>
		</div>
		<div class="details">
			<button type="button" class="magicai-btn magicai--ban-chatbot-ip" data-ip="%3$s">%11$s</button>
		</div>',
		esc_html__( 'User Details', 'magicai-wp' ),
		esc_html__( 'IP', 'magicai-wp' ),
		$user_ip,
		esc_html__( 'User', 'magicai-wp' ),
		$user_name ?? '-',
		esc_html__( 'Conversation', 'magicai-wp' ),
		esc_html__( 'Message', 'magicai-wp' ),
		count($data),
		esc_html__( 'Last Activity', 'magicai-wp' ),
		get_post_meta( $chat_id, '_magicai_user_last_activity', true ) ?? '',
		magicai_helper()->check_user_ip_status( $user_ip ) ? esc_html__( 'Unban This IP', 'magicai-wp' ) : esc_html__( 'Ban This IP', 'magicai-wp' )
	);

	wp_send_json( [
		'chat_id' => $chat_id,
		'output' => $html,
		'header' => $html_header
	] );

}

/**
 * Retrieves chat data based on the user's logged-in status or IP address.
 *
 * Determines the user ID if logged in; otherwise, retrieves the IP address.
 * Retrieves the post ID associated with the transient key based on user or IP.
 * Fetches chat data stored in post meta for the retrieved post ID.
 * Checks for limits on the number of chat messages allowed.
 * Sends a JSON success response containing the retrieved chat data, or an error
 * message if the message limit has been reached.
 *
 * @return void Sends a JSON response with retrieved chat data or an error message.
 * 
 * @since 1.0.0
 */
function magicai_chatbot_get_chat_data() {

	$post_id = !empty( $_POST['chat_id'] ) ? intval( $_POST['chat_id'] ) : '';
	$data = get_post_meta( $post_id, '_magicai_messages', true );

	if ( magicai_helper()->check_user_ip_status() ) {
		wp_send_json( [
			'error' => true,
			'message' => esc_html__('You have been banned by the administrator. Contact the administrator for details.', 'magicai-wp'),
		] );
	}

	if ( !MagicAI_RateLimit::instance()->check( 'chatbot' ) ) {
		wp_send_json( [
			'error' => true,
			'message' => esc_html__('You reached ChatBot message limit!', 'magicai-wp'),
		] );
	}
	
	wp_send_json_success( $data );

}

/**
 * Saves chat data including user prompts and system messages to a post meta.
 *
 * Determines whether the user is logged in or retrieves their IP address if not.
 * Retrieves the post ID associated with the transient key based on user or IP.
 * Sanitizes user prompt and system message inputs.
 * Retrieves existing chat data for the post ID or initializes an empty array.
 * Updates the post meta with new chat entries if the post ID is available.
 * Sends a JSON success response containing the updated chat data.
 *
 * @return void Sends a JSON response with updated chat data or empty data if unsuccessful.
 * 
 * @since 1.0.0
 */
function magicai_chatbot_save_chat_data() {

	if ( is_user_logged_in() ) {
		$user_or_ip = get_current_user_id();
	} else {
		$user_or_ip = magicai_helper()->get_ip() ?? $_SERVER['REMOTE_ADDR'];
	}

	$post_id = !empty( $_POST['chat_id'] ) ? intval( $_POST['chat_id'] ) : '';

	$prompt = sanitize_text_field( $_POST['prompt'] );
	$message = $_POST['message'];

	$data = get_post_meta( $post_id, '_magicai_messages', true ) ?? array();

	if ( ! empty( $post_id ) ) {
		$data[] = [ 'role' => 'user', 'content'=> $prompt ];
		$data[] = [ 'role' => 'system', 'content'=> $message ];
		update_post_meta( $post_id, '_magicai_messages', $data );
		update_post_meta( $post_id, '_magicai_user_last_activity', current_time( 'mysql' ) );
	}

	wp_send_json_success( $data );

}

/**
 * Fetches RSS feed data based on the provided URL.
 *
 * This function takes a URL from the $_POST data, sanitizes it, and attempts to fetch
 * the RSS feed data. It parses the fetched data and prepares an HTML output containing
 * options based on the title of each fetched post.
 *
 * @return void This function doesn't return a value directly but sends a JSON response.
 * 
 * @since 1.0.0
 */
function magicai_fetch_rss() {
	
	$URL = !empty( $_POST['url'] ) ? sanitize_url( $_POST['url'] ) : null;

	if ( empty( $URL ) ) {

		MagicAI_Logs::instance()->add_log(
            'error',
            'RSS to Post',
            'URL is invalid!',
        );

		wp_send_json( [
			'error' => true,
			'message' => esc_html__( 'URL is invalid!', 'magicai-wp' ),
		] );
	}

	$data = magicai_helper()->parseRSS( $URL );

	if ( $data ) {
		MagicAI_Logs::instance()->add_log(
			'completion',
			'RSS is Fetched',
			$URL,
		);
		$html = '';
		foreach( $data as $post ) {
			$html .= sprintf( 
				'<option value="%s">%s</option>',
				esc_attr( $post['title'] ),
				esc_html( $post['title'] ),
			);
		}
		wp_send_json( [
			'output' => $html
		] );
	} else {
		MagicAI_Logs::instance()->add_log(
			'error',
			'RSS is Not Fetched',
			'RSS Not Fetched! URL:'. $URL,
		);
		wp_send_json( [
			'error' => true,
			'message' => esc_html__( 'RSS Not Fetched! Please check your URL and validete the RSS!', 'magicai-wp' ),
		] );
	}

}

/**
 * Saves logs when a 'magicai-chatbot' post is created or updated.
 *
 * @param int     $post_id The ID of the post being saved.
 * @param WP_Post $post    The post object.
 * @param bool    $update  Whether this is an existing post being updated or not.
 */
add_action( 'save_post_magicai-chatbot', function( $post_id, $post, $update ) {

	// If an old book is being updated, exit
	if ( $update ) {
		MagicAI_Logs::instance()->add_log(
			'chatbot',
			'ChatBot Template Updated',
			'Template ID: '. $post_id,
		);
		return;
	}

	MagicAI_Logs::instance()->add_log(
		'chatbot',
		'ChatBot Template Created',
		'Template ID: '. $post_id,
	);
}, 10, 3 );


// Hide Gallery images for other users
add_filter( 'ajax_query_attachments_args', 'magicai_show_current_user_attachments' );
function magicai_show_current_user_attachments( $query ) {
	if ( current_user_can( 'manage_options' ) ) {
		return $query;
	}
	$user_id = get_current_user_id();
	$query['author'] = $user_id;
	return $query;
}

add_action('admin_notices', function(){

	if ( ! magicai_helper()->get_option( 'openai_key' ) ) {
		$svg = '<svg style="vertical-align:sub" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12C22 17.5228 17.5228 22 12 22ZM12 20C16.4183 20 20 16.4183 20 12C20 7.58172 16.4183 4 12 4C7.58172 4 4 7.58172 4 12C4 16.4183 7.58172 20 12 20ZM11 15H13V17H11V15ZM11 7H13V13H11V7Z"></path></svg>';

		printf(
			'<div class="notice notice-warning">
				<p>Before use the MagicAI Demo</p>
				<p style="font-size: 18px"> %1$s Please <strong>make sure</strong> to add an active API key in the settings. Have an API key? <a href="%2$s">Add here</a> or <a href="%3$s" target="_blank">Get your free API Key</a></p>
			</div>',
			$svg,
			admin_url( 'admin.php?page=magicai-settings' ),
			esc_url( 'https://platform.openai.com/account/api-keys' )
		);
	}

});