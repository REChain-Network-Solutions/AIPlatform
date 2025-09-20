<?php

namespace App\Http\Controllers\Finance;

use App\Http\Controllers\Controller;
use App\Http\Middleware\Custom\FinanceLicenseMiddleware;
use App\Models\Currency;
use App\Models\Gateways;
use App\Models\GatewayTax;
use App\Services\GatewaySelector;
use App\Services\Payment\Enums\PaymentGatewayEnum;
use App\Services\Payment\Factories\GatewayFactory;
use Exception;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Str;
use RachidLaasri\LaravelInstaller\Repositories\ApplicationStatusRepositoryInterface;

// Controls ALL Payment Gateway actions
class GatewayController extends Controller
{
    public function __construct(
        public ApplicationStatusRepositoryInterface $applicationStatusRepository
    ) {
        $this->middleware(FinanceLicenseMiddleware::class, ['except' => ['paymentGateways']]);
    }

    public function readManageGatewaysPageData(): array
    {
        $activeGateways = PaymentGatewayEnum::activeGateways();
        $gatewaysDbRecords = Gateways::all()->keyBy('code');
        $requiredGatewayData = [];
        foreach ($activeGateways as $gateway) {
            $enum = PaymentGatewayEnum::tryFrom($gateway);
            if (! $enum) {
                continue;
            }
            $definition = $enum->gatewayDefinition();
            $code = $definition['code'];
            $requiredGatewayData[] = [
                'code'      => $code,
                'title'     => $definition['title'],
                'link'      => $definition['link'],
                'available' => $definition['available'],
                'img'       => $definition['img'],
                'whiteLogo' => $definition['whiteLogo'],
                'active'    => $gatewaysDbRecords[$code]['is_active'] ?? 0,
            ];
        }

        return $requiredGatewayData;
    }

    public function getCurrencyOptions($index)
    {
        $returnText = '';
        $currencies = Currency::all();
        foreach ($currencies as $currency) {
            $cindex = $currency->id;
            $country = $this->appendNBSPtoString($currency->country, 41);
            $code = $this->appendNBSPtoString($currency->code, 5);
            $text = $country . $code . $currency->symbol;
            $selected = (int) $index === (int) $cindex ? 'selected' : '';
            $returnText .= '<option value="' . $cindex . '" ' . $selected . ' style=\'font-family: "Courier New", Courier, monospace;\' >' . $text . '</option>';
        }

        return $returnText;
    }

    public function appendNBSPtoString($stringForAppend, $charCount) // Fills given string with &nbsp; at the end. Used in Country select tag.
    {
        $length = Str::length($stringForAppend);
        $remainingCharcount = $charCount - $length;
        if ($remainingCharcount < 1) {
            return $stringForAppend;
        }

        $newString = $stringForAppend;
        for ($i = 1; $i <= $remainingCharcount; $i++) {
            $newString .= '&nbsp;';
        }

        return $newString;
    }

    public function paymentGateways()
    {
        $gateways = $this->readManageGatewaysPageData();

        return view(
            'panel.admin.finance.gateways.index', [
                'gateways' => $gateways,
                'view'     => view($this->applicationStatusRepository->financePage(), [
                    'gateways' => $gateways,
                ])->render(),
            ]
        );
    }

    // Settings page of gateways in Admin Panel
    public function gatewaySettings($code)
    {
        $activeGateways = PaymentGatewayEnum::activeGateways();
        if (! in_array($code, $activeGateways, true)) {
            abort(404);
        }

        $gatewayDbRecord = Gateways::query()->where('code', $code)->first();
        if (empty($gatewayDbRecord)) {
            $gatewayDbRecord = new Gateways;
            $gatewayDbRecord->code = $code;
            $gatewayDbRecord->is_active = 0;
            $gatewayDbRecord->currency = '124'; // Default currency for Stripe - USD
            $gatewayDbRecord->save();
        }
        $currencies = $this->getCurrencyOptions($gatewayDbRecord->currency);
        $gatewayDefinition = PaymentGatewayEnum::tryFrom($code)?->gatewayDefinition();
        $taxes = GatewayTax::query()
            ->where('gateway_id', $gatewayDbRecord->id)
            ->get();

        return view('panel.admin.finance.gateways.settings', compact('gatewayDbRecord', 'currencies', 'gatewayDefinition', 'taxes'));
    }

    public function countryTaxEnabled($code)
    {
        $settings = Gateways::query()->where('code', $code)->firstOrFail();

        $settings->update([
            'country_tax_enabled' => ! $settings->country_tax_enabled,
        ]);

        return response()->json([
            'message' => 'Setting updated successfully.',
        ]);
    }

    public function gatewaySettingsSave(Request $request) // Save settings of gateway in Admin Panel
    {
        if ($request->code) {
            if (! in_array($request->code, PaymentGatewayEnum::activeGateways(), true)) {
                abort(404);
            }
        } else {
            abort(404);
        }
        // return 404 error if the system currency is not the same as the gateway currency
        if ($request->code == 'paystack' && $request->currency != currency()->id) {
            return back()->with(['message' => __('Paystack default currency not the same with the system default currency.'), 'type' => 'error']);
        }

        DB::beginTransaction();
        $gw_settings = Gateways::where('code', $request->code)->first();
        if ($gw_settings != null) {
            if ($request->is_active == 'on') {
                $gw_settings->is_active = 1;
            } else {
                $gw_settings->is_active = 0;
            }

            if ($request->automate_tax == 'on') {
                $gw_settings->automate_tax = 1;
            } else {
                $gw_settings->automate_tax = 0;
            }
            $propertiesToUpdate = [
                'title', 'currency', 'currency_locale', 'live_client_id', 'live_client_secret',
                'live_app_id', 'sandbox_client_id', 'sandbox_client_secret', 'sandbox_app_id',
                'base_url', 'sandbox_url', 'mode', 'bank_account_other', 'bank_account_details',
            ];
            foreach ($propertiesToUpdate as $property) {
                if (isset($request->$property)) {
                    $gw_settings->$property = $request->$property ?? $gw_settings->$property;
                }
            }
            $gw_settings->save();

            if ($gw_settings->is_active == 1) {
                try {
                    if (PaymentGatewayEnum::isRefactored($request->code)) {
                        GatewayFactory::make(PaymentGatewayEnum::tryFrom($request->code))->saveAllProducts();
                    } else {
                        $temp = GatewaySelector::selectGateway($request->code)::saveAllProducts(); // Update all product ids' and create new price ids'
                    }
                } catch (Exception $ex) {
                    DB::rollBack();
                    Log::error("GatewayController::gatewaySettingsSave()\n" . $ex->getMessage());

                    return back()->with(['message' => $ex->getMessage(), 'type' => 'error']);
                }
            }
        } else {
            $settings = new Gateways;
            $settings->code = $request->code;
            $settings->is_active = 0;
            $settings->currency = '124'; // Default currency for Stripe - USD
            $settings->save();
        }
        DB::commit();

        return back()->with(['message' => __('Product ID and Price ID of all membership plans are generated.'), 'type' => 'success']);
    }

    public function gatewaySettingsTaxSave(Request $request) // Save settings of gateway in Admin Panel
    {
        $gw_settings = Gateways::query()->where('code', $request->code)->firstOrFail();

        // return 404 error if the system currency is not the same as the gateway currency
        if ($request->code == 'paystack' && $request->currency != currency()->id) {
            return back()->with(['message' => __('Paystack default currency not the same with the system default currency.'), 'type' => 'error']);
        }

        if ($request->code == 'cryptomus') {
            GatewayTax::query()
                ->updateOrCreate([
                    'gateway_id'   => $gw_settings->id,
                    'country_code' => $request->get('country_code'),
                ], [
                    'tax' => $request->get('tax'),
                ]);
        } else {
            DB::beginTransaction();

            $gw_settings->tax = $request->tax ?? $gw_settings->tax;
            $gw_settings->save();

            DB::commit();
        }

        return back()->with(['message' => __('Tax saved succesfully.'), 'type' => 'success']);
    }

    public function gatewaySettingsTaxDelete($id)
    {
        $gatewayTax = GatewayTax::query()->findOrFail($id);

        $gatewayTax->delete();

        return back()->with([
            'type'    => 'success',
            'message' => __('Tax deleted successfully.'),
        ]);
    }

    public function gatewayData($code): ?array
    {
        return PaymentGatewayEnum::tryFrom($code)?->gatewayDefinition();
    }

    public static function checkGatewayWebhooks(): void
    {
        $host = $_SERVER['HTTP_HOST'];
        if ($host !== 'localhost:8000' && $host !== '127.0.0.1:8000') {
            $gateways = Gateways::all();
            foreach ($gateways as $gateway) {
                if ($gateway->webhook_id == null) {
                    $tmp = GatewaySelector::selectGateway($gateway->code)::createWebhook();
                }
            }
            Log::info('All gateways are checked for webhooks.');
        } else {
            Log::info('Webhooks are not available on localhost. Skipping checkGatewayWebhooks()...');
        }
    }
}
