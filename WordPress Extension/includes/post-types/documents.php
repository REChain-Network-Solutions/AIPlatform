<?php 

register_post_type( 'magicai-documents',
    array(
        'labels' => array(
            'name' => esc_html__( 'Documents', 'magicai-wp' ),
            'search_items' => esc_html__( 'Search Documents', 'magicai-wp' ),
        ),
    'public' => true,
    'has_archive' => false,
    'exclude_from_search' => false,
    'capability_type' => 'page',
    'show_in_menu' => 'magicai',
    )
);
