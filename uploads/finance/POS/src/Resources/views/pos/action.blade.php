@permission('pos show')
<div class="action-btn me-2">
    <a href="{{ route('pos.show',\Crypt::encrypt($pos->id)) }}"
        data-bs-whatever="{{ __('Common case Details') }}"
        data-title="{{ __('View') }}" class="mx-3 btn bg-warning btn-sm  align-items-center"
        data-bs-original-title="{{ __('View') }}" data-bs-toggle="tooltip"
        data-bs-placement="top"><i class="ti ti-eye text-white"></i>
    </a>
</div>
@endpermission

