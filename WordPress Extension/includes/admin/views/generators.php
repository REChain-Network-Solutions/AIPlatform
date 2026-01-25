<div class="wrap magicai-page-wrap">
    <span class="magicai-loader"></span>
    <h1><?php echo esc_html( get_admin_page_title() ); ?></h1>
    <p class="magicai-page-description custom-generator"><?php esc_html_e( 'Generate any kind of content in seconds.', 'magicai-wp' ); ?></p>
    <p class="magicai-page-description image-generator"><?php esc_html_e( 'Genereate stunning visuals in seconds.', 'magicai-wp' ); ?></p>
    <p class="magicai-page-description code-generator"><?php esc_html_e( 'Print code like a programmer with simple definitions.', 'magicai-wp' ); ?></p>
    <p class="magicai-page-description product-generator"><?php esc_html_e( 'Generate SEO-optimized product content in a matter of seconds.', 'magicai-wp' ); ?></p>
    <p class="magicai-page-description post-generator"><?php esc_html_e( 'Generate SEO-optimized blog content in a matter of seconds.', 'magicai-wp' ); ?></p>
    <p class="magicai-page-description yt-post-generator"><?php esc_html_e( 'Simply turn your Youtube videos into Blog post.', 'magicai-wp' ); ?></p>
    <p class="magicai-page-description rss-post-generator"><?php esc_html_e( 'Generate unique blog posts with RSS Feed.', 'magicai-wp' ); ?></p>
    
    <!-- tab navs -->
    <div id="magicai-settings-tabs-wrapper" class="magicai-nav-tab-wrapper nav-tab-wrapper">
        <a id="magicai-settings-tab-custom-generator" class="nav-tab" href="#tab-custom-generator">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M15 5.25C16.7949 5.25 18.25 3.79493 18.25 2H19.75C19.75 3.79493 21.2051 5.25 23 5.25V6.75C21.2051 6.75 19.75 8.20507 19.75 10H18.25C18.25 8.20507 16.7949 6.75 15 6.75V5.25ZM4 7C4 5.89543 4.89543 5 6 5H13V3H6C3.79086 3 2 4.79086 2 7V17C2 19.2091 3.79086 21 6 21H18C20.2091 21 22 19.2091 22 17V12H20V17C20 18.1046 19.1046 19 18 19H6C4.89543 19 4 18.1046 4 17V7Z"></path></svg>
            <?php esc_html_e( 'Custom', 'magicai-wp' ); ?>
        </a>

        <a id="magicai-settings-tab-image-generator" class="nav-tab" href="#tab-image-generator">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M2.9918 21C2.44405 21 2 20.5551 2 20.0066V3.9934C2 3.44476 2.45531 3 2.9918 3H21.0082C21.556 3 22 3.44495 22 3.9934V20.0066C22 20.5552 21.5447 21 21.0082 21H2.9918ZM20 15V5H4V19L14 9L20 15ZM20 17.8284L14 11.8284L6.82843 19H20V17.8284ZM8 11C6.89543 11 6 10.1046 6 9C6 7.89543 6.89543 7 8 7C9.10457 7 10 7.89543 10 9C10 10.1046 9.10457 11 8 11Z"></path></svg>
            <?php esc_html_e( 'Image Generator', 'magicai-wp' ); ?>
        </a>

        <a id="magicai-settings-tab-code-generator" class="nav-tab" href="#tab-code-generator">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M24 12L18.3431 17.6569L16.9289 16.2426L21.1716 12L16.9289 7.75736L18.3431 6.34315L24 12ZM2.82843 12L7.07107 16.2426L5.65685 17.6569L0 12L5.65685 6.34315L7.07107 7.75736L2.82843 12ZM9.78845 21H7.66009L14.2116 3H16.3399L9.78845 21Z"></path></svg>
            <?php esc_html_e( 'Code Generator', 'magicai-wp' ); ?>
        </a>

        <a id="magicai-settings-tab-product-generator" class="nav-tab" href="#tab-product-generator">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M4.00436 6.41662L0.761719 3.17398L2.17593 1.75977L5.41857 5.00241H20.6603C21.2126 5.00241 21.6603 5.45012 21.6603 6.00241C21.6603 6.09973 21.6461 6.19653 21.6182 6.28975L19.2182 14.2898C19.0913 14.7127 18.7019 15.0024 18.2603 15.0024H6.00436V17.0024H17.0044V19.0024H5.00436C4.45207 19.0024 4.00436 18.5547 4.00436 18.0024V6.41662ZM6.00436 7.00241V13.0024H17.5163L19.3163 7.00241H6.00436ZM5.50436 23.0024C4.67593 23.0024 4.00436 22.3308 4.00436 21.5024C4.00436 20.674 4.67593 20.0024 5.50436 20.0024C6.33279 20.0024 7.00436 20.674 7.00436 21.5024C7.00436 22.3308 6.33279 23.0024 5.50436 23.0024ZM17.5044 23.0024C16.6759 23.0024 16.0044 22.3308 16.0044 21.5024C16.0044 20.674 16.6759 20.0024 17.5044 20.0024C18.3328 20.0024 19.0044 20.674 19.0044 21.5024C19.0044 22.3308 18.3328 23.0024 17.5044 23.0024Z"></path></svg>
            <?php esc_html_e( 'Product Generator', 'magicai-wp' ); ?>
        </a>

        <a id="magicai-settings-tab-post-generator" class="nav-tab" href="#tab-post-generator">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M20 22H4C3.44772 22 3 21.5523 3 21V3C3 2.44772 3.44772 2 4 2H20C20.5523 2 21 2.44772 21 3V21C21 21.5523 20.5523 22 20 22ZM19 20V4H5V20H19ZM7 6H11V10H7V6ZM7 12H17V14H7V12ZM7 16H17V18H7V16ZM13 7H17V9H13V7Z"></path></svg>
            <?php esc_html_e( 'Post Generator', 'magicai-wp' ); ?>
        </a>

        <a id="magicai-settings-tab-yt-post-generator" class="nav-tab" href="#tab-yt-post-generator">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M19.6069 6.99482C19.5307 6.69695 19.3152 6.47221 19.0684 6.40288C18.6299 6.28062 16.501 6 12.001 6C7.50098 6 5.37252 6.28073 4.93225 6.40323C4.68776 6.47123 4.4723 6.69593 4.3951 6.99482C4.2863 7.41923 4.00098 9.19595 4.00098 12C4.00098 14.804 4.2863 16.5808 4.3954 17.0064C4.47126 17.3031 4.68676 17.5278 4.93251 17.5968C5.37252 17.7193 7.50098 18 12.001 18C16.501 18 18.6299 17.7194 19.0697 17.5968C19.3142 17.5288 19.5297 17.3041 19.6069 17.0052C19.7157 16.5808 20.001 14.8 20.001 12C20.001 9.2 19.7157 7.41923 19.6069 6.99482ZM21.5442 6.49818C22.001 8.28 22.001 12 22.001 12C22.001 12 22.001 15.72 21.5442 17.5018C21.2897 18.4873 20.547 19.2618 19.6056 19.5236C17.8971 20 12.001 20 12.001 20C12.001 20 6.10837 20 4.39637 19.5236C3.45146 19.2582 2.70879 18.4836 2.45774 17.5018C2.00098 15.72 2.00098 12 2.00098 12C2.00098 12 2.00098 8.28 2.45774 6.49818C2.71227 5.51273 3.45495 4.73818 4.39637 4.47636C6.10837 4 12.001 4 12.001 4C12.001 4 17.8971 4 19.6056 4.47636C20.5505 4.74182 21.2932 5.51636 21.5442 6.49818ZM10.001 15.5V8.5L16.001 12L10.001 15.5Z"></path></svg>
            <?php esc_html_e( 'AI Youtube', 'magicai-wp' ); ?>
        </a>

        <a id="magicai-settings-tab-rss-post-generator" class="nav-tab" href="#tab-rss-post-generator">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M3 3C12.9411 3 21 11.0589 21 21H18C18 12.7157 11.2843 6 3 6V3ZM3 10C9.07513 10 14 14.9249 14 21H11C11 16.5817 7.41828 13 3 13V10ZM3 17C5.20914 17 7 18.7909 7 21H3V17Z"></path></svg>
            <?php esc_html_e( 'RSS to Post', 'magicai-wp' ); ?>
        </a>
    </div>

    <!-- tab contents -->
    <?php
        include __DIR__ . '/tabs/post.php';
        include __DIR__ . '/tabs/rss-post.php';
        include __DIR__ . '/tabs/yt-post.php';
        include __DIR__ . '/tabs/product.php';
        include __DIR__ . '/tabs/custom.php';
        include __DIR__ . '/tabs/code.php';
        include __DIR__ . '/tabs/image.php';
    ?>

</div>
