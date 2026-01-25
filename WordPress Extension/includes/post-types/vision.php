<?php 

register_post_type( 'magicai-vision',
    array(
        'labels' => array(
            'name' => esc_html__( 'MagicAI Vision', 'magicai-wp' ),
        ),
    'public' => true,
    'has_archive' => false,
    'exclude_from_search' => false,
    'capability_type' => 'page',
    'show_in_menu' => false,
    )
);
