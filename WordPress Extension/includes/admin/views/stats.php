<div class="wrap magicai-page-wrap">
    <span class="magicai-loader"></span>
    <h1><?php esc_html_e( get_admin_page_title() ); ?></h1>
    <p><?php esc_html_e( 'Statistics for the last 7 days.' ); ?></p>

    <div class="magicai-stats">

        <div class="magicai-stats--col">
            <h2><?php esc_html_e( 'Generated Content', 'magicai-wp' ); ?></h2>

            <?php
                $stats_results = MagicAI_Stats::instance()->get_results();
                $text_data = $image_data = array();

                // Get the last 7 days
                $last_seven_days = array();
                for ($i = 0; $i < 7; $i++) {
                    $last_seven_days[] = date('d.m', strtotime("-$i days"));
                    $text_data[] = $stats_results[date('d.m', strtotime("-$i days"))]['text']['count'] ?? 0;
                    $image_data[] = $stats_results[date('d.m', strtotime("-$i days"))]['image']['count'] ?? 0;
                }
                $last_seven_days = array_reverse( $last_seven_days );
                $text_data = array_reverse( $text_data );
                $image_data = array_reverse( $image_data );

                $content_data = [
                    'labels' => $last_seven_days,
                    'datasets' => [
                        [
                            'label' => __( 'Word', 'magicai-wp' ),
                            'data' => $text_data,
                            'borderWidth' => 1,
                        ],
                        [
                            'label' => __( 'Image', 'magicai-wp' ),
                            'data' => $image_data,
                            'borderWidth' => 1,
                        ],
                    ]
                ];
            ?>
            <canvas id="content" data-data="<?php echo esc_attr( wp_json_encode( $content_data ) ); ?>"></canvas>
        </div>

        <div class="magicai-stats--col">
            <h2><?php esc_html_e( 'Generator Usage', 'magicai-wp' ); ?></h2>

            <?php
            $stats_results = MagicAI_Stats::instance()->get_results( 'generator' );

                 $generator_data = [
                    'labels' => [
                        __( 'Post Generator', 'magicai-wp' ),
                        __( 'Product Generator', 'magicai-wp' ),
                        __( 'Custom Generator', 'magicai-wp' ),
                        __( 'Code Generator', 'magicai-wp' ),
                        __( 'RSS Generator', 'magicai-wp' ),
                        __( 'YouTube Generator', 'magicai-wp' ),
                        __( 'Assistant', 'magicai-wp' ),
                        __( 'Image DALL-E', 'magicai-wp' ),
                        __( 'Image SD', 'magicai-wp' ),
                    ],
                    'datasets' => [
                        [
                            'data' => [
                                $stats_results['post_generator']['count'] ?? 0,
                                $stats_results['product_generator']['count'] ?? 0,
                                $stats_results['custom_generator']['count'] ?? 0,
                                $stats_results['code_generator']['count'] ?? 0,
                                $stats_results['rss']['count'] ?? 0,
                                $stats_results['youtube']['count'] ?? 0,
                                $stats_results['assistant']['count'] ?? 0,
                                $stats_results['dall-e']['count'] ?? 0,
                                $stats_results['sd']['count'] ?? 0,
                            ],
                        ],
                    ]
                ];
            ?>
            <canvas id="generator" data-data="<?php echo esc_attr( wp_json_encode( $generator_data ) ); ?>"></canvas>
        </div>

        <div class="magicai-stats--col">
            <h2><?php esc_html_e( 'Frontend / WP Dasboard Usage', 'magicai-wp' ); ?></h2>

            <?php
            $stats_results = MagicAI_Stats::instance()->get_results( 'is_frontend' );

                 $generator_data = [
                    'labels' => [
                        __( 'Frontend', 'magicai-wp' ),
                        __( 'WP Dashboard', 'magicai-wp' ),
                    ],
                    'datasets' => [
                        [
                            'data' => [
                                $stats_results['frontend'] ?? 0,
                                $stats_results['backend'] ?? 0,
                            ],
                        ],
                    ]
                ];
            ?>
            <canvas id="frontend_usage" data-data="<?php echo esc_attr( wp_json_encode( $generator_data ) ); ?>"></canvas>
        </div>

        <div>
            <?php _e( 'More statistics will be available here soon.', 'magicai-wp' ); ?>
        </div>

    </div>


</div>
