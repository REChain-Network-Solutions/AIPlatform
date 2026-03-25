<?php

namespace Modules\Quotes\Http\Controllers;

use App\Http\Controllers\AccountBaseController;
use Illuminate\Http\Request;
use Illuminate\Support\Str;
use Modules\Quotes\Entities\Quote;
use Modules\Quotes\Entities\QuoteItem;
use Modules\Quotes\Http\Requests\StoreQuoteRequest;

class QuoteController extends AccountBaseController
{
    public function index()
    {
        $quotes = Quote::latest()->paginate(20);
        $this->pageTitle = 'Quotes & Estimates';
        return view('quotes::quotes.index', compact('quotes'));
    }

    public function create()
    {
        $this->pageTitle = 'Create Estimate / Quote';
        $currency = config('quotes.default_currency', 'USD');
        return view('quotes::quotes.create', compact('currency'));
    }

    public function store(StoreQuoteRequest $request)
    {
        $data = $request->validated();

        $number = $this->generateQuoteNumber();

        $quote = Quote::create([
            'number' => $number,
            'client_id' => $data['client_id'] ?? null,
            'currency' => $data['currency'],
            'status' => 'draft',
            'valid_until' => $data['valid_until'] ?? null,
            'notes' => $data['notes'] ?? null,
            'subtotal' => 0,
            'tax_total' => 0,
            'grand_total' => 0,
        ]);

        $subtotal = 0; $tax_total = 0;
        foreach ($data['items'] as $it) {
            $qty = (float)$it['qty'];
            $price = (float)$it['unit_price'];
            $rate = isset($it['tax_rate']) ? (float)$it['tax_rate'] : 0;
            $line = round($qty * $price, 2);
            $tax = round($line * $rate, 2);
            QuoteItem::create([
                'quote_id' => $quote->id,
                'item_id' => $it['item_id'] ?? null,
                'description' => $it['description'],
                'qty' => $qty,
                'unit_price' => $price,
                'tax_rate' => $rate,
                'line_total' => $line,
            ]);
            $subtotal += $line;
            $tax_total += $tax;
        }

        $quote->subtotal = $subtotal;
        $quote->tax_total = $tax_total;
        $quote->grand_total = $subtotal + $tax_total;
        $quote->save();

        return redirect()->route('quotes.show', $quote->id)->with('success', 'Quote created');
    }

    public function show($id)
    {
        $quote = Quote::with('items')->findOrFail($id);
        $this->pageTitle = 'Quote ' . $quote->number;
        return view('quotes::quotes.show', compact('quote'));
    }

    public function edit($id)
    {
        $quote = Quote::with('items')->findOrFail($id);
        $this->pageTitle = 'Edit Estimate / Quote ' . $quote->number;
        return view('quotes::quotes.edit', compact('quote'));
    }

    public function update(StoreQuoteRequest $request, $id)
    {
        $quote = Quote::with('items')->findOrFail($id);
        $data = $request->validated();

        // Replace items
        $quote->items()->delete();
        $subtotal = 0; $tax_total = 0;
        foreach ($data['items'] as $it) {
            $qty = (float)$it['qty'];
            $price = (float)$it['unit_price'];
            $rate = isset($it['tax_rate']) ? (float)$it['tax_rate'] : 0;
            $line = round($qty * $price, 2);
            $tax = round($line * $rate, 2);
            QuoteItem::create([
                'quote_id' => $quote->id,
                'item_id' => $it['item_id'] ?? null,
                'description' => $it['description'],
                'qty' => $qty,
                'unit_price' => $price,
                'tax_rate' => $rate,
                'line_total' => $line,
            ]);
            $subtotal += $line;
            $tax_total += $tax;
        }

        $quote->update([
            'client_id' => $data['client_id'] ?? null,
            'currency' => $data['currency'],
            'valid_until' => $data['valid_until'] ?? null,
            'notes' => $data['notes'] ?? null,
            'subtotal' => $subtotal,
            'tax_total' => $tax_total,
            'grand_total' => $subtotal + $tax_total,
        ]);

        return redirect()->route('quotes.show', $quote->id)->with('success', 'Quote updated');
    }

    public function destroy($id)
    {
        $quote = Quote::findOrFail($id);
        $quote->items()->delete();
        $quote->delete();
        return redirect()->route('quotes.index')->with('success', 'Quote deleted');
    
        public function searchItems(Request $request)
    {
        $currency = (string) $request->query('currency', '');
        $priceListId = (string) $request->query('price_list_id', '');
        $term = trim((string)$request->query('term', ''));
        $out = [];

        // Helper to pick the first non-null/non-empty property
        $pick = function ($obj, array $candidates, $default = null) {
            foreach ($candidates as $c) {
                if (isset($obj->{$c}) && $obj->{$c} !== '' && $obj->{$c} !== null) {
                    return $obj->{$c};
                }
            }
            return $default;
        };

        // 1) Try Items module with common column names
        try {
            if (class_exists('Modules\\Items\\Entities\\Item')) {
                $q = \Modules\Items\Entities\Item::query();
                if ($term !== '') {
                    $q->where(function($qq) use($term){
                        $qq->where('name', 'like', '%'.$term.'%')
                           ->orWhere('title', 'like', '%'.$term.'%')
                           ->orWhere('sku', 'like', '%'.$term.'%')
                           ->orWhere('description', 'like', '%'.$term.'%');
                    });
                }
                foreach ($q->limit(20)->get() as $it) {
                    $label = (string) $pick($it, ['name','title','sku','description'], 'Item #'.$it->id);
                    $unit  = (float)  $pick($it, ['price','unit_price','sale_price','amount','rate','cost'], 0);
                    // tax_rate direct or tax_percent/vat_percent -> convert to decimal
                    $taxVal = $pick($it, ['tax_rate','vat_rate'], null);
                    if ($taxVal === null) {
                        $taxPercent = $pick($it, ['tax_percent','vat_percent'], null);
                        $taxVal = $taxPercent !== null ? ((float)$taxPercent)/100.0 : 0.0;
                    } else {
                        $taxVal = (float) $taxVal;
                    }

                    $out[] = [
                        'label' => $label,
                        'value' => $label,
                        'unit_price' => round($unit, 2),
                        'tax_rate' => round($taxVal, 4),
                    ];
                }
            }
        } catch (\Throwable $e) { /* ignore */ }

        // 2) Fallback to PriceListItem
        if (empty($out)) {
            $pl = \Modules\Quotes\Entities\PriceListItem::query()->with('priceList');
            if ($priceListId !== '') { $pl->where('price_list_id', (int)$priceListId); }
            if ($currency !== '') { $pl->whereHas('priceList', function($q) use($currency){ $q->where('currency', $currency); }); }
            if ($term !== '') {
                $pl->where('item_name', 'like', '%'.$term.'%');
            }
            foreach ($pl->limit(20)->get() as $pi) {
                $out[] = [
                    'label' => $pi->item_name,
                    'value' => $pi->item_name,
                    'unit_price' => (float) $pi->unit_price,
                    'tax_rate' => (float) $pi->tax_rate,
                ];
            }
        }

        return response()->json($out);
    
    public function publicShow($id, Request $request)
    {
        $quote = Quote::with('items')->findOrFail($id);
        // Build accept/reject signed routes (re-sign within the view time)
        $acceptUrl = \URL::temporarySignedRoute('quotes.public.action', now()->addDays(30), ['id' => $id, 'action' => 'accept']);
        $rejectUrl = \URL::temporarySignedRoute('quotes.public.action', now()->addDays(30), ['id' => $id, 'action' => 'reject']);
        return view('quotes::quotes.public', compact('quote','acceptUrl','rejectUrl'));
    }

    
    public function publicAction($id, $action, Request $request)
    {
        $quote = Quote::with('items')->findOrFail($id);
        if ($action === 'accept') {
            $auto = (bool) config('quotes.auto_convert_on_accept', false);
            $quote->status = 'accepted';
            $quote->accepted_at = now();
            $quote->rejected_at = null;
            $quote->save();

            $invoiceId = null;
            if ($auto && class_exists('Modules\EInvoice\Entities\Invoice')) {
                $inv = new \Modules\EInvoice\Entities\Invoice();
                $inv->client_id = $quote->client_id;
                $inv->currency = $quote->currency;
                $inv->status = 'draft';
                $inv->notes = 'Auto-converted from accepted quote ' . $quote->number;
                $inv->subtotal = $quote->subtotal;
                $inv->tax_total = $quote->tax_total;
                $inv->grand_total = $quote->grand_total;
                $inv->save();
                $invoiceId = $inv->id;

                if (class_exists('Modules\EInvoice\Entities\InvoiceItem')) {
                    foreach ($quote->items as $qi) {
                        \Modules\EInvoice\Entities\InvoiceItem::create([
                            'invoice_id' => $inv->id,
                            'description' => $qi->description,
                            'qty' => $qi->qty,
                            'unit_price' => $qi->unit_price,
                            'line_total' => $qi->line_total,
                        ]);
                    }
                }
            }

            // Staff notification emails
            try {
                $emails = array_filter(array_map('trim', explode(',', (string) config('quotes.notify.emails', ''))));
                if (!empty($emails)) {
                    foreach ($emails as $em) {
                        \Mail::to($em)->send(new \\Modules\\Quotes\\Mail\\QuoteAccepted($quote, $invoiceId));
                    }
                }
            } catch (\Throwable $e) { /* ignore mail failures */ }

            // Webhook
            try {
                $url = (string) config('quotes.webhook.accept_url', '');
                if (!empty($url)) {
                    $payload = [
                        'event' => 'quote.accepted',
                        'quote_id' => $quote->id,
                        'quote_number' => $quote->number,
                        'client_id' => $quote->client_id,
                        'currency' => $quote->currency,
                        'subtotal' => (float) $quote->subtotal,
                        'tax_total' => (float) $quote->tax_total,
                        'grand_total' => (float) $quote->grand_total,
                        'accepted_at' => optional($quote->accepted_at)->toAtomString(),
                        'invoice_id' => $invoiceId,
                    ];
                    $secret = (string) config('quotes.webhook.secret', '');
                    $sig = hash_hmac('sha256', json_encode($payload), $secret ?: 'no-secret');
                    if (class_exists('Illuminate\\Support\\Facades\\Http')) {
                        \Illuminate\Support\Facades\Http::withHeaders(['X-Quotes-Signature' => $sig])->post($url, $payload);
                    } else {
                        // cURL fallback
                        $ch = curl_init($url);
                        curl_setopt_array($ch, [
                            CURLOPT_RETURNTRANSFER => true,
                            CURLOPT_POST => true,
                            CURLOPT_HTTPHEADER => ['Content-Type: application/json', 'X-Quotes-Signature: ' . $sig],
                            CURLOPT_POSTFIELDS => json_encode($payload),
                            CURLOPT_TIMEOUT => 5,
                        ]);
                        curl_exec($ch);
                        curl_close($ch);
                    }
                }
            } catch (\Throwable $e) { /* ignore webhook failures */ }

            return view('quotes::quotes.public_result', ['quote' => $quote, 'result' => 'accepted', 'invoice_id' => $invoiceId]);
        } elseif ($action === 'reject') {
    
            $quote->status = 'rejected';
            $quote->rejected_at = now();
            $quote->accepted_at = null;
            $quote->save();
            return view('quotes::quotes.public_result', ['quote' => $quote, 'result' => 'rejected']);
        }
        abort(400, 'Invalid action');
    

    private function generateQuoteNumber(): string
    {
        $prefix = (string) config('quotes.number_prefix', 'Q-');
        $pattern = (string) config('quotes.number_pattern', '{PREFIX}{YYYY}-{NNNNN}');
        $series = (string) config('quotes.sequence_series', 'default');
        $year = (int) now()->format('Y');

        // Lock row for update to avoid race
        $seq = \Modules\Quotes\Entities\QuoteSequence::where('series', $series)->where('year', $year)->lockForUpdate()->first();
        if (!$seq) {
            $seq = \Modules\Quotes\Entities\QuoteSequence::create(['series' => $series, 'year' => $year, 'next_number' => 1]);
        }
        $num = $seq->next_number;
        $seq->next_number = $num + 1;
        $seq->save();

        $replacements = [
            '{PREFIX}' => $prefix,
            '{YYYY}' => (string) $year,
            '{YY}' => substr((string)$year, -2),
            '{MM}' => now()->format('m'),
            '{DD}' => now()->format('d'),
        ];

        // Handle N padding tokens
        $padded = [
            '{NN}' => str_pad((string)$num, 2, '0', STR_PAD_LEFT),
            '{NNN}' => str_pad((string)$num, 3, '0', STR_PAD_LEFT),
            '{NNNN}' => str_pad((string)$num, 4, '0', STR_PAD_LEFT),
            '{NNNNN}' => str_pad((string)$num, 5, '0', STR_PAD_LEFT),
        ];

        $out = strtr($pattern, $replacements);
        $out = strtr($out, $padded);
        return $out;
    }

}
