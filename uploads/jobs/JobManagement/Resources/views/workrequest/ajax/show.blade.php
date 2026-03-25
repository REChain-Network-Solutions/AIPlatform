<div class="card border-0 invoice">
    <!-- CARD BODY START -->
    <div class="card-body">
        <div class="invoice-table-wrapper">
            <table width="100%">
                <tr class="inv-logo-heading">
                    <td class="font-weight-bold f-21 text-dark text-uppercase mt-4 mt-lg-0 mt-md-0">
                        @lang('engineerings::modules.wr')</td>
                </tr>
                <tr class="inv-num">
                    <td class="f-14 text-dark">
                        <table class="inv-num-date text-dark f-13 mt-3 border">
                            <tr>
                                <td class="border-0 f-w-500">
                                    @lang('engineerings::app.menu.ticketID')</td>
                                <td class="border-0">: <a
                                        href="{{ route('complaint.show', $wr->ticket->complaint_number) }}">#{{ $wr->ticket->id }}</a>
                                </td>
                            </tr>
                            <tr class="">
                                <td class="border-0 f-w-500">
                                    @lang('engineerings::app.menu.ticketSubject')</td>
                                <td class="border-0">: {{ $wr->ticket->subject }}</td>
                            </tr>
                        </table>
                    </td>
                    <td align="right">
                        <table class="inv-num-date text-dark f-13 mt-3">
                            <tr>
                                <td class="bg-light-grey border-right-0 f-w-500">
                                    @lang('engineerings::app.menu.noWR')</td>
                                <td class="border-left-0">#{{ $wr->wr_no }}</td>
                            </tr>
                            <tr>
                                <td class="bg-light-grey border-right-0 f-w-500">@lang('engineerings::app.menu.date')</td>
                                <td class="border-left-0">
                                    {{ \Carbon\Carbon::createFromFormat('Y-m-d H:i:s', $wr->check_time)->format('Y-m-d | H:i') }}
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
                <tr>
                    <td height="20"></td>
                </tr>
            </table>
            <div class="row">
                <div class="col-md-12">
                    <table width="100%" class="inv-desc d-none d-lg-table d-md-table">
                        <tr>
                            <td class="f-14 text-black" width="10%">
                                <p class="line-height mb-0">
                                    <span class="text-grey text-capitalize">@lang('engineerings::app.menu.assignTo')</span>
                                </p>
                            </td>
                            <td class="f-14 text-black" width="75%">
                                <p class="line-height mb-0">
                                    <span class="text-black text-capitalize">:
                                        {{ $wr->user->name ?? '--' }}
                                    </span>
                                </p>
                            </td>
                        </tr>
                        <tr>
                            <td class="f-14 text-black" width="10%">
                                <p class="line-height mb-0">
                                    <span class="text-grey text-capitalize">@lang('engineerings::app.menu.remark')</span>
                                </p>
                            </td>
                            <td class="f-14 text-black" width="75%">
                                <p class="line-height mb-0">
                                    <span class="text-black text-capitalize">: {{ ucwords($wr->remark) }} </span>
                                </p>
                            </td>
                        </tr>
                        <tr>
                            <td class="f-14 text-black" width="10%">
                                <p class="line-height mb-0">
                                    <span class="text-grey text-capitalize">@lang('engineerings::app.menu.tenant')</span>
                                </p>
                            </td>
                            <td class="f-14 text-black" width="75%">
                                <p class="line-height mb-0">
                                    <span class="text-black text-capitalize">:
                                        {{ $wr->charge_by_tenant === 1 ? 'Yes' : ($wr->charge_by_tenant === 0 ? 'No' : null) }}</span>
                                </p>
                            </td>
                        </tr>
                        <tr>
                            <td class="f-14 text-black" width="10%">
                                <p class="line-height mb-0">
                                    <span class="text-grey text-capitalize">@lang('complaint::app.menu.area')</span>
                                </p>
                            </td>
                            <td class="f-14 text-black" width="75%">
                                <p class="line-height mb-0">
                                    <span class="text-black text-capitalize">:
                                        {{ ucwords($wr->house->area->area_name) }}</span>
                                </p>
                            </td>
                        </tr>
                        <tr>
                            <td class="f-14 text-black" width="10%">
                                <p class="line-height mb-0">
                                    <span class="text-grey text-capitalize">@lang('complaint::app.menu.houses')</span>
                                </p>
                            </td>
                            <td class="f-14 text-black" width="75%">
                                <p class="line-height mb-0">
                                    <span class="text-black text-capitalize">:
                                        {{ ucwords($wr->house->house_name) }}</span>
                                </p>
                            </td>
                        </tr>
                        <tr>
                            <td class="f-14 text-black" width="10%">
                                <p class="line-height mb-0">
                                    <span class="text-grey text-capitalize">@lang('engineerings::app.menu.problem')</span>
                                </p>
                            </td>
                            <td class="f-14 text-black" width="75%">
                                <p class="line-height mb-0">
                                    <span class="text-black text-capitalize">:
                                        {{ ucwords($wr->problem) }}</span>
                                </p>
                            </td>
                        </tr>
                        <td colspan="2" class="mt-4">
                            <table class="inv-detail f-14 table-responsive-sm mt-4" width="100%">
                                @foreach ($wr->items as $item)
                                    <tr class="i-d-heading text-dark-grey font-weight-bold">
                                        <td width="35%" class="border bg-light-grey">@lang('Item')</td>
                                        <td width="12%" class="border bg-light-grey" align="right">
                                            @lang('engineerings::app.menu.qty')
                                        </td>
                                        <td width="16%" class="border bg-light-grey" align="right">
                                            @lang('engineerings::app.menu.harga')
                                        </td>
                                        <td width="17%" class="border bg-light-grey" align="right">
                                            @lang('engineerings::app.menu.tax')
                                        </td>
                                        <td width="20%" class="border bg-light-grey" align="right">
                                            @lang('engineerings::app.menu.total')
                                        </td>
                                    </tr>
                                    <tr class="text-dark font-weight-semibold f-13 border">
                                        <td>{{ ucwords($item->item->name) }}</td>
                                        <td align="right">
                                            {{ $item->qty }} <input type="hidden" name="item_name"
                                                value="{{ $item->qty }}">
                                        </td>
                                        <td align="right">
                                            {{ $item->harga }} <input type="hidden" name="harga"
                                                value="{{ $item->harga }}">
                                        </td>
                                        <td align="right">
                                            @if ($item->tax != 0)
                                                {{ $item->tax . '%' }}
                                            @endif <input type="hidden" name="tax"
                                                value="{{ $item->tax }}">
                                        </td>
                                        <td align="right"><span class="jumlah">0.00</span></td>
                                    </tr>
                                @endforeach

                                @foreach ($wr->services as $service)
                                    <tr class="i-d-heading text-dark-grey font-weight-bold">
                                        <td width="35%" class="border bg-light-grey">@lang('Service')</td>
                                        <td width="12%" class="border bg-light-grey" align="right">
                                            @lang('engineerings::app.menu.qty')
                                        </td>
                                        <td width="16%" class="border bg-light-grey" align="right">
                                            @lang('engineerings::app.menu.harga')
                                        </td>
                                        <td width="17%" class="border bg-light-grey" align="right">
                                            @lang('engineerings::app.menu.tax')
                                        </td>
                                        <td width="20%" class="border bg-light-grey" align="right">
                                            @lang('engineerings::app.menu.total')
                                        </td>
                                    </tr>
                                    <tr class="text-dark font-weight-semibold f-13 border">
                                        <td>{{ ucwords($service->service->name) }}</td>
                                        <td align="right">
                                            {{ $service->qty }} <input type="hidden" name="item_name"
                                                value="{{ $service->qty }}">
                                        </td>
                                        <td align="right">
                                            {{ $service->harga }} <input type="hidden" name="harga"
                                                value="{{ $service->harga }}">
                                        </td>
                                        <td align="right">
                                            @if ($service->tax != 0)
                                                {{ $service->tax . '%' }}
                                            @endif <input type="hidden" name="tax"
                                                value="{{ $service->tax }}">
                                        </td>
                                        <td align="right"><span class="jumlah">0.00</span></td>
                                    </tr>
                                @endforeach
                                <tr class="d-none d-md-table-row d-lg-table-row f-14">
                                    <td colspan="4" class="dash-border-top border bg-amt-grey" align="right">
                                        <b>@lang('modules.invoices.total')</b>
                                    </td>
                                    <td class="dash-border-top border"><span class="jumlah-total">0.00</span></td>
                                </tr>
                            </table>
                        </td>
                        </tr>
                    </table>
                </div>
                <div class="col-md-4">
                    <table class="inv-num-date text-dark f-13 mt-3 border">
                        <tr>
                            <td class="bg-light-grey f-w-500">
                                @lang('engineerings::app.menu.foto')</td>
                        </tr>
                        <tr>
                            <td>
                                @if ($url == null)
                                    no image
                                @else
                                    <img src="{{ $url }}" alt="{{ $wr->problem }}">
                                @endif
                            </td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>
    </div>
    <!-- CARD BODY END -->
    <!-- CARD FOOTER START -->
    <div class="card-footer bg-white border-0 d-flex justify-content-start py-0 py-lg-4 py-md-4 mb-4 mb-lg-3 mb-md-3 ">
        <div class="d-flex">
            <div class="inv-action mr-3 mr-lg-3 mr-md-3 dropup">
                <button class="dropdown-toggle btn-primary" type="button" id="dropdownMenuButton"
                    data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">@lang('app.action')
                    <span><i class="fa fa-chevron-up f-15"></i></span>
                </button>
                <!-- DROPDOWN - INFORMATION -->
                <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton" tabindex="0">
                    @if ($wr->status_wo != 1)
                        <li>
                            <a class="dropdown-item f-14 text-dark"
                                href="{{ route('engineerings.edit', [$wr->id]) }}">
                                <i class="fa fa-edit f-w-500 mr-2 f-11"></i> @lang('app.edit')
                            </a>
                        </li>
                    @endif
                    <li>
                        <a class="dropdown-item f-14 text-dark delete-invoice" href="javascript:;"
                            data-invoice-id="{{ $wr->id }}">
                            <i class="fa fa-trash f-w-500 mr-2 f-11"></i> @lang('app.delete')
                        </a>
                    </li>
                    <li>
                        <a class="dropdown-item f-14 text-dark"
                            href="{{ route('engineerings.download', [$wr->id]) }}">
                            <i class="fa fa-download f-w-500 mr-2 f-11"></i> @lang('app.download')
                        </a>
                    </li>

                </ul>
            </div>
            <x-forms.button-cancel :link="route('engineerings.index')" class="border-0 mr-3">@lang('app.cancel')
            </x-forms.button-cancel>
        </div>


    </div>
    <!-- CARD FOOTER END -->
</div>
<script>
    $(document).ready(function() {
        $('input[name^="item_name"], input[name^="harga"]').each(function() {
            hitungTotalBaris($(this).closest('tr'));
        });
        hitungTotalSemua();
    });

    function hitungTotalBaris(row) {
        const qty = Number(row.find('input[name^="item_name"]').val());
        const costPerItem = Number(row.find('input[name^="harga"]').val());
        const tax = Number(row.find('input[name^="tax"]').val());
        const total = qty * costPerItem;
        let jml_tax = 0; // Menginisialisasi nilai jml_tax dengan 0
        let jml_total = 0; // Menginisialisasi nilai jml_total dengan 0

        // Memeriksa apakah tax yang dipilih adalah 0 atau bukan, dan melakukan perhitungan sesuai dengan itu
        if (tax !== 0) {
            jml_tax = (tax / 100) * total;
        }

        jml_total = jml_tax + total;
        // const formattedTotal = formatRupiah(jml_total);

        row.find('span.jumlah').text(jml_total);
    }

    function hitungTotalSemua() {
        // Dapatkan semua elemen span dengan kelas "jumlah"
        const semuaJumlah = Array.from(document.querySelectorAll('span.jumlah'));

        // Ubah nilai "innerHTML" dari semua elemen span menjadi nilai numerik, lalu jumlahkan dengan metode "reduce()"
        const total = semuaJumlah.map(jumlah => parseInt(jumlah.innerHTML.replace(/\D/g, ''))).reduce((
            total, jumlah) => total + jumlah, 0);

        // Ubah nilai dari elemen span dengan kelas "jumlah-total" menjadi hasil perhitungan di atas
        document.querySelector('span.jumlah-total').textContent = total;
    }


    $('body').on('click', '.delete-invoice', function() {
        var id = $(this).data('invoice-id');
        Swal.fire({
            title: "@lang('messages.sweetAlertTitle')",
            text: "@lang('messages.recoverRecord')",
            icon: 'warning',
            showCancelButton: true,
            focusConfirm: false,
            confirmButtonText: "@lang('messages.confirmDelete')",
            cancelButtonText: "@lang('app.cancel')",
            customClass: {
                confirmButton: 'btn btn-primary mr-3',
                cancelButton: 'btn btn-secondary'
            },
            showClass: {
                popup: 'swal2-noanimation',
                backdrop: 'swal2-noanimation'
            },
            buttonsStyling: false
        }).then((result) => {
            if (result.isConfirmed) {
                var token = "{{ csrf_token() }}";

                var url = "{{ route('engineerings.destroy', ':id') }}";
                url = url.replace(':id', id);

                $.easyAjax({
                    type: 'POST',
                    url: url,
                    blockUI: true,
                    data: {
                        '_token': token,
                        '_method': 'DELETE'
                    },
                    success: function(response) {
                        if (response.status == "success") {
                            window.location.href = "{{ route('engineerings.index') }}";
                        }
                    }
                });
            }
        });
    });
</script>
