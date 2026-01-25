<?php 

register_post_type( 'magicai-attachments',
    array(
        'labels' => array(
            'name' => esc_html__( 'Attachments', 'magicai-wp' ),
            'search_items' => esc_html__( 'Search Attachments', 'magicai-wp' ),
        ),
    'public' => true,
    'has_archive' => false,
    'exclude_from_search' => false,
    'capability_type' => 'page',
    )
);
