<?php

namespace App\Http\Controllers\Market;

use App\Domains\Marketplace\Repositories\Contracts\ExtensionRepositoryInterface;
use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Arr;
use Illuminate\Validation\ValidationException;

class MarketPlaceController extends Controller
{
    public function __construct(
        public ExtensionRepositoryInterface $extensionRepository
    ) {}

    public function deleteCoupon(): RedirectResponse
    {
        $data = $this->extensionRepository->deleteCoupon();

        return back()->with([
            'type'    => 'success',
            'message' => trans('Coupon removed successfully.'),
            'status'  => 'success',
        ]);
    }

    public function cartCoupon(Request $request): RedirectResponse
    {
        $request->validate(['coupon_code' => 'required|string|max:255']);

        $data = $this->extensionRepository->storeCoupon($request['coupon_code']);

        if ($data['status'] === 'success') {
            return back()->with([
                'type'    => 'success',
                'message' => trans('Coupon applied successfully.'),
                'status'  => 'success',
            ]);
        }

        throw ValidationException::withMessages([
            'coupon_code' => $data['message'],
        ]);
    }

    public function cart(Request $request)
    {
        $items = $this->extensionRepository->cart();

        return view('panel.admin.marketplace.cart', compact('items'));
    }

    public function addDelete(Request $request, int $id)
    {
        $extension = $this->extensionRepository->findId($id);

        $url = data_get($extension, 'routes.cart-add-or-delete');

        if ($url) {

            $response = $this->extensionRepository->request(
                'post',
                '',
                [],
                $url
            );

            if ($response->ok()) {
                $count = is_array($response->json('data')) ? count($response->json('data')) : 0;

                return response()->json([
                    'cart_html' => '',
                    'itemCount' => $count,
                    'iconId'    => $id . '-icon',
                    'message'   => $response->json('message'),
                    'action'    => $response->json('action'),
                ]);
            }
        }

        return response()->json([
            'cart_html' => '',
            'itemCount' => 0,
            'iconId'    => $id . '-icon',
            'message'   => trans('Bir hata oluştu'),
            'action'    => 'error',
        ]);
    }

    public function index()
    {
        $items = $this->extensionRepository->supportExtensions();

        $banners = $this->extensionRepository->banners();

        $subscription = $this->extensionRepository->subscription()->json();

        $cart = data_get($this->extensionRepository->cart(), 'data', []);

        $cartExists = Arr::pluck($cart, 'extension.id');

        $paymentStatus = request('payment_status');

        return view('panel.admin.marketplace.index', compact('items', 'banners', 'subscription', 'cart', 'cartExists', 'paymentStatus'));
    }

    public function extension($slug)
    {
        $paidExtensions = $this->extensionRepository->paidExtensions();

        $item = $this->extensionRepository->findSupport($slug);

        $marketSubscription = $this->extensionRepository->subscription()->json();

        $cart = data_get($this->extensionRepository->cart(), 'data', []);

        $cartExists = Arr::pluck($cart, 'extension.id');

        if (! $item) {
            return to_route('dashboard.admin.marketplace.index')->with('error', 'Extension not found.');
        }

        if ($item['type'] === 'bundle') {
            return view('panel.admin.marketplace.bundle', compact('item', 'marketSubscription', 'cart', 'cartExists', 'paidExtensions'));
        }

        return view('panel.admin.marketplace.show', compact('item', 'marketSubscription', 'cart', 'cartExists', 'paidExtensions'));
    }

    public function licensedExtension()
    {
        $cart = data_get($this->extensionRepository->cart(), 'data', []);

        $items = $this->extensionRepository->licensed(
            $this->extensionRepository->supportExtensions()
        );

        return view('panel.admin.marketplace.licensed', compact('items', 'cart'));
    }

    public function buyExtension($slug)
    {
        abort('404');

        $item = $this->extensionRepository->find($slug);

        if (! $item) {
            return to_route('dashboard.admin.marketplace.index')->with('error', 'Extension not found.');
        }

        $response = $this->extensionRepository->request(
            'get',
            '',
            [],
            $item['routes']['paymentJson'] . '?version=' . $this->extensionRepository->appVersion()
        );

        if ($response->ok()) {
            $data = $response->json('data');

            return view('panel.admin.marketplace.payment', compact('item', 'data'));
        }

        if (! $item) {
            return to_route('dashboard.admin.marketplace.index')->with('error', 'Extension not found.');
        }
    }

    public function extensionActivate(Request $request, string $token)
    {

        cache()->forget('check_license_domain_' . $request->getHost());

        $data = Helper::decodePaymentToken($token);

        $item = $this->extensionRepository->find($data['slug']);

        return view('panel.admin.marketplace.activate', [
            'item'    => $item,
            'token'   => $token,
            'success' => $request->get('redirect_status') == 'succeeded',
        ]);
    }
}
