<div class="wrap">
    <span class="magicai-loader"></span>
    <div class="magicai-chat">
        <div class="magicai-chat--search magicai-form">
            <div class="form-field">
                <input type="text" onkeyup="filter_chat_list(this)" placeholder="<?php esc_attr_e( 'Search chat...', 'magicai-wp' ); ?>">
            </div>
        </div>
        <div class="magicai-chat--header">
            <div class="magicai-chat--header-title">
                <?php esc_html_e( 'New Chat', 'magicai-wp' ); ?>
            </div>
            <div class="magicai-chat--header-options">
                <!-- <div class="magicai-switch">
                    <span><?php esc_html_e( 'Real-Time data', 'magicai-wp' ); ?></span>
                    <input type="checkbox" id="web_search" name="web_search" />
                    <label for="web_search"></label>
                </div> -->
                <div class="magicai-chat--toggle">
                    <span class="dashicons dashicons-ellipsis"></span>
                    <span class="dashicons dashicons-no-alt"></span>
                </div>
            </div>
        </div>
        <div class="magicai-chat--list">
            <div class="magicai-chat--list-before"></div>
            <div class="magicai-chat--search-mobile magicai-form">
                <div class="form-field">
                    <input type="text" onkeyup="filter_chat_list(this)" placeholder="<?php esc_attr_e( 'Search chat...', 'magicai-wp' ); ?>">
                </div>
            </div>
            <button class="magicai-chat--new" data-type="vision">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M14 3V5H4V18.3851L5.76282 17H20V10H22V18C22 18.5523 21.5523 19 21 19H6.45455L2 22.5V4C2 3.44772 2.44772 3 3 3H14ZM19 3V0H21V3H24V5H21V8H19V5H16V3H19Z"></path></svg>
                <span><?php esc_html_e( 'New Chat', 'magicai-wp' ); ?></span>
            </button>
            <?php magicai_helper()->get_chat_list( $atts, $type = 'vision' ); ?>
        </div>
        <div class="magicai-chat--messages">
            <div class="magicai-chat--message-list">
                <div class="magicai-chat--message ai">
					<div class="text">
						<img width="32" src="<?php echo esc_url( MAGICAI_URL . 'assets/img/logo.svg' ); ?>">
                        <?php esc_html_e( 'Seamlessly upload any image you want to explore or discuss and get insightful conversations.', 'magicai-wp' ); ?>
					</div>
				</div>
            </div>
            <form id="chat-form" class="magicai-chat--form" data-type="vision">
                <button type="button" class="upload">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18"><path d="M21 15V18H24V20H21V23H19V20H16V18H19V15H21ZM21.0082 3C21.556 3 22 3.44495 22 3.9934V13H20V5H4V18.999L14 9L17 12V14.829L14 11.8284L6.827 19H14V21H2.9918C2.44405 21 2 20.5551 2 20.0066V3.9934C2 3.44476 2.45531 3 2.9918 3H21.0082ZM8 7C9.10457 7 10 7.89543 10 9C10 10.1046 9.10457 11 8 11C6.89543 11 6 10.1046 6 9C6 7.89543 6.89543 7 8 7Z" fill="currentColor"></path></svg>
                </button>
                <input type="hidden" name>
                <input type="text" name="prompt" id="prompt" placeholder="<?php esc_attr_e( 'Type and hit...', 'magicai-wp' ) ?>" required autocomplete="off">
                <button type="submit" class="start">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 24 24" width="18" height="18"><path d="M3.5 1.3457C3.58425 1.3457 3.66714 1.36699 3.74096 1.4076L22.2034 11.562C22.4454 11.695 22.5337 11.9991 22.4006 12.241C22.3549 12.3241 22.2865 12.3925 22.2034 12.4382L3.74096 22.5925C3.499 22.7256 3.19497 22.6374 3.06189 22.3954C3.02129 22.3216 3 22.2387 3 22.1544V1.8457C3 1.56956 3.22386 1.3457 3.5 1.3457ZM5 4.38261V11.0001H10V13.0001H5V19.6175L18.8499 12.0001L5 4.38261Z"></path></svg>
                </button>
                <button type="button" class="stop">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 24 24" width="18" height="18"><path d="M12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12C22 17.5228 17.5228 22 12 22ZM12 20C16.4183 20 20 16.4183 20 12C20 7.58172 16.4183 4 12 4C7.58172 4 4 7.58172 4 12C4 16.4183 7.58172 20 12 20ZM9 9H15V15H9V9Z"></path></svg>
                </button>
            </form>
        </div>
    </div>
</div>