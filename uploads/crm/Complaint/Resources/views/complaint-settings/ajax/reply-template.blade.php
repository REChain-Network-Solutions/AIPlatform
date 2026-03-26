<div class="table-responsive p-20">
    <x-table class="table-bordered">
        <x-slot name="thead">
            <th>@lang('app.name')</th>
            <th class="text-right">@lang('app.action')</th>
        </x-slot>

        @forelse($templates as $key=> $template)
            <tr class="row{{ $template->id }}">
                <td>{{ mb_ucwords($template->reply_heading) }}</td>
                <td class="text-right">
                    <div class="task_view">
                        <a class="task_view_more d-flex align-items-center justify-content-center edit-template" href="javascript:;" data-template-id="{{ $template->id }}" >
                            <i class="fa fa-edit icons mr-2"></i>  @lang('app.edit')
                        </a>
                    </div>
                    <div class="task_view">
                        <a class="task_view_more d-flex align-items-center justify-content-center delete-template" href="javascript:;" data-template-id="{{ $template->id }}" >
                            <i class="fa fa-trash icons mr-2"></i> @lang('app.delete')
                        </a>
                    </div>
                </td>
            </tr>
        @empty
            <tr>
                <td colspan="3">
                    <x-cards.no-record icon="list" :message="__('messages.noTemplateFound')" />
                </td>
            </tr>
        @endforelse
    </x-table>
</div>
