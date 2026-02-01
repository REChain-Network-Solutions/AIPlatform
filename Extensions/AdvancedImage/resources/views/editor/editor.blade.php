<div
    class="lqd-adv-img-editor-editor group/editor pointer-events-none invisible fixed inset-0 z-2 flex min-h-screen bg-background opacity-0 transition-all"
    :class="{
        'opacity-0': currentView !== 'editor',
        'invisible': currentView !== 'editor',
        'pointer-events-none': currentView !== 'editor',
        'active': currentView === 'editor',
        'sidebar-collapsed': sidebarCollapsed
    }"
>
    @include('advanced-image::editor.editor-toolbar', ['tools' => $tools, 'primary_tool_keys' => $primary_tool_keys])
    @include('advanced-image::editor.editor-canvas')
    @include('advanced-image::editor.editor-sidebar', ['tools' => $tools])

    <div
        class="lqd-adv-img-editor-notif fixed bottom-8 start-1/2 z-10 -translate-x-1/2 text-3xs/4"
        x-ref="editorNotif"
    >
        <div
            class="flex items-center gap-2.5 rounded-lg bg-heading-foreground/90 p-2.5 text-background/95 backdrop-blur-lg backdrop-saturate-150 lg:whitespace-nowrap"
            x-show="showNotif"
            x-transition
        >
            <svg
                width="16"
                height="16"
                viewBox="0 0 16 16"
                fill="currentColor"
                xmlns="http://www.w3.org/2000/svg"
            >
                <path
                    d="M7.4375 11.5625H8.56246V7.24998H7.4375V11.5625ZM7.99998 5.96634C8.17162 5.96634 8.31549 5.90829 8.43159 5.79219C8.54769 5.67609 8.60574 5.53222 8.60574 5.36058C8.60574 5.18896 8.54769 5.04509 8.43159 4.92898C8.31549 4.81288 8.17162 4.75483 7.99998 4.75483C7.82834 4.75483 7.68448 4.81288 7.56838 4.92898C7.45228 5.04509 7.39423 5.18896 7.39423 5.36058C7.39423 5.53222 7.45228 5.67609 7.56838 5.79219C7.68448 5.90829 7.82834 5.96634 7.99998 5.96634ZM8.00124 15.125C7.01579 15.125 6.08951 14.938 5.22241 14.564C4.3553 14.19 3.60104 13.6824 2.95963 13.0413C2.3182 12.4001 1.81041 11.6462 1.43624 10.7795C1.06208 9.91277 0.875 8.98669 0.875 8.00124C0.875 7.01579 1.062 6.08951 1.436 5.22241C1.81 4.3553 2.31756 3.60104 2.95869 2.95963C3.59983 2.3182 4.35376 1.81041 5.22048 1.43624C6.08719 1.06208 7.01328 0.875 7.99873 0.875C8.98418 0.875 9.91045 1.062 10.7776 1.436C11.6447 1.81 12.3989 2.31756 13.0403 2.95869C13.6818 3.59983 14.1896 4.35376 14.5637 5.22048C14.9379 6.08719 15.125 7.01328 15.125 7.99873C15.125 8.98418 14.938 9.91045 14.564 10.7776C14.19 11.6447 13.6824 12.3989 13.0413 13.0403C12.4001 13.6818 11.6462 14.1896 10.7795 14.5637C9.91277 14.9379 8.98669 15.125 8.00124 15.125ZM7.99998 14C9.67498 14 11.0937 13.4187 12.2562 12.2562C13.4187 11.0937 14 9.67498 14 7.99998C14 6.32498 13.4187 4.90623 12.2562 3.74373C11.0937 2.58123 9.67498 1.99998 7.99998 1.99998C6.32498 1.99998 4.90623 2.58123 3.74373 3.74373C2.58123 4.90623 1.99998 6.32498 1.99998 7.99998C1.99998 9.67498 2.58123 11.0937 3.74373 12.2562C4.90623 13.4187 6.32498 14 7.99998 14Z"
                />
            </svg>
            <span>
                @lang('Image Saved to') <u>@lang('Public Gallery')</u>
            </span>
        </div>
    </div>
</div>
