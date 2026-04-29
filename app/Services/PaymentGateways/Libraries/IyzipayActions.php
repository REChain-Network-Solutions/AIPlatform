<?php

namespace App\Services\PaymentGateways\Libraries;

use App\Http\Controllers\Controller;
use App\Models\Currency;
use Illuminate\Support\Facades\Log;
use Iyzipay\Model\Address;
use Iyzipay\Model\BasketItem;
use Iyzipay\Model\Buyer;
use Iyzipay\Model\CheckoutForm;
use Iyzipay\Model\CheckoutFormInitialize;
use Iyzipay\Model\Customer;
use Iyzipay\Model\Locale;
use Iyzipay\Model\Subscription\RetrieveList;
use Iyzipay\Model\Subscription\SubscriptionCancel;
use Iyzipay\Model\Subscription\SubscriptionCheckoutForm;
use Iyzipay\Model\Subscription\SubscriptionCustomer;
use Iyzipay\Model\Subscription\SubscriptionDetails;
use Iyzipay\Model\Subscription\SubscriptionPricingPlan;
use Iyzipay\Model\Subscription\SubscriptionProduct;
use Iyzipay\Options;
use Iyzipay\Request\CreateCheckoutFormInitializeRequest;
use Iyzipay\Request\RetrieveCheckoutFormRequest;
use Iyzipay\Request\Subscription\SubscriptionCancelRequest;
use Iyzipay\Request\Subscription\SubscriptionCreateCheckoutFormRequest;
use Iyzipay\Request\Subscription\SubscriptionCreateCustomerRequest;
use Iyzipay\Request\Subscription\SubscriptionCreatePricingPlanRequest;
use Iyzipay\Request\Subscription\SubscriptionCreateProductRequest;
use Iyzipay\Request\Subscription\SubscriptionDeleteCustomerRequest;
use Iyzipay\Request\Subscription\SubscriptionDeletePricingPlanRequest;
use Iyzipay\Request\Subscription\SubscriptionDeleteProductRequest;
use Iyzipay\Request\Subscription\SubscriptionDetailsRequest;
use Iyzipay\Request\Subscription\SubscriptionListProductsRequest;
use Iyzipay\Request\Subscription\SubscriptionRetrieveCustomerRequest;
use Iyzipay\Request\Subscription\SubscriptionRetrievePricingPlanRequest;
use Iyzipay\Request\Subscription\SubscriptionRetrieveProductRequest;
use Iyzipay\Request\Subscription\SubscriptionUpdateCustomerRequest;
use Iyzipay\Request\Subscription\SubscriptionUpdatePricingPlanRequest;
use Iyzipay\Request\Subscription\SubscriptionUpdateProductRequest;

// a class that makes subscription and payments with iyzipay
class IyzipayActions // extends Controller
{
    // api keys
    private $config;

    // locale
    private $locale;

    // currency
    private $currency;

    public function __construct($apiKey, $apiSecretKey, $baseUrl, $locale = Locale::TR, $currency = \Iyzipay\Model\Currency::TL)
    {
        $this->config = new Options;
        $this->config->setApiKey($apiKey);
        $this->config->setSecretKey($apiSecretKey);
        $this->config->setBaseUrl($baseUrl);
        $this->locale = $locale;
        $this->currency = $currency;

        // Log::info("iyzipayActions constructor called : " . $apiKey . " " . $apiSecretKey . " " . $baseUrl . " " . $locale . " " . $currency);
    }

    // get config
    public function getConfig()
    {
        return $this->config;
    }

    // get locale
    public function getLocale()
    {
        return $this->locale;
    }

    // generate random string with given length
    public function generateRandomString($length = 12)
    {
        return substr(str_shuffle(str_repeat($x = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', ceil($length / strlen($x)))), 1, $length);
    }

    // generate random number with given length
    public function generateRandomNumber($length = 12)
    {
        return substr(str_shuffle(str_repeat($x = '0123456789', ceil($length / strlen($x)))), 1, $length);
    }

    // ************************************************

    // ******* SUBSCRIPTION CUSTOMER ACTIONS **********

    // ************************************************

    // create customer for \Iyzipay\Model\Customer
    public function createCustomer($request)
    {
        $customer = new Customer;
        $customer->setName($request->name ?? '');
        $customer->setSurname($request->surname ?? '');
        $customer->setGsmNumber($request->gsmNumber ?? '');
        $customer->setEmail($request->email ?? '');
        $customer->setIdentityNumber($request->identityNumber ?? '');
        $customer->setShippingContactName($request->shippingContactName ?? '');
        $customer->setShippingCity($request->shippingCity ?? '');
        $customer->setShippingCountry($request->shippingCountry ?? '');
        $customer->setShippingAddress($request->shippingAddress ?? '');
        $customer->setShippingZipCode($request->shippingZipCode ?? '');
        $customer->setBillingContactName($request->billingContactName ?? '');
        $customer->setBillingCity($request->billingCity ?? '');
        $customer->setBillingCountry($request->billingCountry ?? '');
        $customer->setBillingAddress($request->billingAddress ?? '');
        $customer->setBillingZipCode($request->billingZipCode ?? '');

        return $customer;
    }

    // create a subscription customer for \Iyzipay\Request\Subscription\SubscriptionCreateCustomerRequest
    public function createSubscriptionCustomer($request)
    {
        $requestSubscriptionCustomer = new SubscriptionCreateCustomerRequest;
        $requestSubscriptionCustomer->setLocale($this->locale);
        $requestSubscriptionCustomer->setConversationId($this->generateRandomNumber());

        $customer = new Customer;
        $customer->setName($request->name ?? '');
        $customer->setSurname($request->surname ?? '');
        $customer->setGsmNumber($request->gsmNumber ?? '');
        $customer->setEmail($request->email ?? '');
        $customer->setIdentityNumber($request->identityNumber ?? '');
        $customer->setShippingContactName($request->shippingContactName ?? '');
        $customer->setShippingCity($request->shippingCity ?? '');
        $customer->setShippingCountry($request->shippingCountry ?? '');
        $customer->setShippingAddress($request->shippingAddress ?? '');
        $customer->setShippingZipCode($request->shippingZipCode ?? '');
        $customer->setBillingContactName($request->billingContactName ?? '');
        $customer->setBillingCity($request->billingCity ?? '');
        $customer->setBillingCountry($request->billingCountry ?? '');
        $customer->setBillingAddress($request->billingAddress ?? '');
        $customer->setBillingZipCode($request->billingZipCode ?? '');

        $requestSubscriptionCustomer->setCustomer($customer);

        $subscriptionCustomer = SubscriptionCustomer::create($requestSubscriptionCustomer, $this->config);

        return $subscriptionCustomer;
    }

    // retrieve a subscription customer for \Iyzipay\Request\Subscription\SubscriptionRetrieveCustomerRequest
    public function retrieveSubscriptionCustomer($request)
    {
        $requestSubscriptionCustomer = new SubscriptionRetrieveCustomerRequest;
        $requestSubscriptionCustomer->setLocale($this->locale);
        $requestSubscriptionCustomer->setConversationId($this->generateRandomNumber());
        $requestSubscriptionCustomer->setCustomerReferenceCode($request->customerReferenceCode);

        $subscriptionCustomer = SubscriptionCustomer::retrieve($requestSubscriptionCustomer, $this->config);

        return $subscriptionCustomer;
    }

    // update a subscription customer for \Iyzipay\Request\Subscription\SubscriptionUpdateCustomerRequest
    public function updateSubscriptionCustomer($request)
    {
        $requestSubscriptionCustomer = new SubscriptionUpdateCustomerRequest;
        $requestSubscriptionCustomer->setLocale($this->locale);
        $requestSubscriptionCustomer->setConversationId($this->generateRandomNumber());
        $requestSubscriptionCustomer->setCustomerReferenceCode($request->customerReferenceCode);

        $customer = new Customer;
        $customer->setName($request->name ?? '');
        $customer->setSurname($request->surname ?? '');
        $customer->setGsmNumber($request->gsmNumber ?? '');
        $customer->setEmail($request->email ?? '');
        $customer->setIdentityNumber($request->identityNumber ?? '');
        $customer->setShippingContactName($request->shippingContactName ?? '');
        $customer->setShippingCity($request->shippingCity ?? '');
        $customer->setShippingCountry($request->shippingCountry ?? '');
        $customer->setShippingDistrict($request->shippingDistrict ?? '');
        $customer->setShippingAddress($request->shippingAddress ?? '');
        $customer->setShippingZipCode($request->shippingZipCode ?? '');
        $customer->setBillingContactName($request->billingContactName ?? '');
        $customer->setBillingCity($request->billingCity ?? '');
        $customer->setBillingCountry($request->billingCountry ?? '');
        $customer->setBillingDistrict($request->billingDistrict ?? '');
        $customer->setBillingAddress($request->billingAddress ?? '');
        $customer->setBillingZipCode($request->billingZipCode ?? '');

        $requestSubscriptionCustomer->setCustomer($customer);

        $subscriptionCustomer = SubscriptionCustomer::update($requestSubscriptionCustomer, $this->config);

        return $subscriptionCustomer;
    }

    // delete a subscription customer for \Iyzipay\Request\Subscription\SubscriptionDeleteCustomerRequest
    public function deleteSubscriptionCustomer($request)
    {
        $requestSubscriptionCustomer = new SubscriptionDeleteCustomerRequest;
        $requestSubscriptionCustomer->setLocale($this->locale);
        $requestSubscriptionCustomer->setConversationId($this->generateRandomNumber());
        $requestSubscriptionCustomer->setCustomerReferenceCode($request->customerReferenceCode);

        $subscriptionCustomer = SubscriptionCustomer::delete($requestSubscriptionCustomer, $this->config);

        return $subscriptionCustomer;
    }

    // ************************************************

    // ******** SUBSCRIPTION PRODUCT ACTIONS **********

    // ************************************************

    // create a subscription product for \Iyzipay\Request\Subscription\SubscriptionCreateProductRequest
    public function createSubscriptionProduct($request)
    {

        $requestSubscriptionProduct = new SubscriptionCreateProductRequest;
        $requestSubscriptionProduct->setLocale($this->locale);
        $requestSubscriptionProduct->setConversationId($this->generateRandomNumber());
        $requestSubscriptionProduct->setName($request->name);
        $requestSubscriptionProduct->setDescription($request->description ?? '');

        $subscriptionProduct = SubscriptionProduct::create($requestSubscriptionProduct, $this->config);

        return $subscriptionProduct;
    }

    // retrieve a subscription product for \Iyzipay\Request\Subscription\SubscriptionRetrieveProductRequest
    public function retrieveSubscriptionProduct($request)
    {
        $requestSubscriptionProduct = new SubscriptionRetrieveProductRequest;
        $requestSubscriptionProduct->setLocale($this->locale);
        $requestSubscriptionProduct->setConversationId($this->generateRandomNumber());
        $requestSubscriptionProduct->setProductReferenceCode($request->productReferenceCode);

        $subscriptionProduct = SubscriptionProduct::retrieve($requestSubscriptionProduct, $this->config);

        return $subscriptionProduct;
    }

    // update a subscription product for \Iyzipay\Request\Subscription\SubscriptionUpdateProductRequest
    public function updateSubscriptionProduct($request)
    {
        $requestSubscriptionProduct = new SubscriptionUpdateProductRequest;
        $requestSubscriptionProduct->setLocale($this->locale);
        $requestSubscriptionProduct->setConversationId($this->generateRandomNumber());
        $requestSubscriptionProduct->setProductReferenceCode($request->productReferenceCode);
        $requestSubscriptionProduct->setName($request->name);
        $requestSubscriptionProduct->setDescription($request->description ?? '');

        $subscriptionProduct = SubscriptionProduct::update($requestSubscriptionProduct, $this->config);

        return $subscriptionProduct;
    }

    // delete a subscription product for \Iyzipay\Request\Subscription\SubscriptionDeleteProductRequest
    public function deleteSubscriptionProduct($request)
    {
        $requestSubscriptionProduct = new SubscriptionDeleteProductRequest;
        $requestSubscriptionProduct->setLocale($this->locale);
        $requestSubscriptionProduct->setConversationId($this->generateRandomNumber());
        $requestSubscriptionProduct->setProductReferenceCode($request->productReferenceCode);

        $subscriptionProduct = SubscriptionProduct::delete($requestSubscriptionProduct, $this->config);

        return $subscriptionProduct;
    }

    // get all subscription products for \Iyzipay\Request\Subscription\SubscriptionListProductsRequest
    public function listSubscriptionProducts($request)
    {
        $requestSubscriptionProduct = new SubscriptionListProductsRequest;
        $requestSubscriptionProduct->setPage($request->itemPage ?? 1);
        $requestSubscriptionProduct->setCount($request->itemCount ?? 50);

        $subscriptionProduct = RetrieveList::products($requestSubscriptionProduct, $this->config);

        return $subscriptionProduct;
    }

    // *****************************************************************

    // ************ SUBSCRIPTION PRICING PLAN ACTIONS ******************

    // *****************************************************************

    // create a subscription pricing plan for \Iyzipay\Request\Subscription\SubscriptionCreatePricingPlanRequest
    public function createSubscriptionPricingPlan($request)
    {
        $requestSubscriptionPricingPlan = new SubscriptionCreatePricingPlanRequest;
        $requestSubscriptionPricingPlan->setLocale($this->locale);
        $requestSubscriptionPricingPlan->setConversationId($this->generateRandomNumber());
        $requestSubscriptionPricingPlan->setProductReferenceCode($request->productReferenceCode);
        $requestSubscriptionPricingPlan->setName($request->name ?? '');
        $requestSubscriptionPricingPlan->setCurrencyCode($this->currency);
        $requestSubscriptionPricingPlan->setPrice($request->price ?? 1);
        $requestSubscriptionPricingPlan->setPaymentInterval($request->paymentInterval ?? 'MONTHLY');
        $requestSubscriptionPricingPlan->setPaymentIntervalCount($request->paymentIntervalCount ?? 1);
        $requestSubscriptionPricingPlan->setPlanPaymentType($request->paymentType ?? 'RECURRING');
        $requestSubscriptionPricingPlan->setTrialPeriodDays($request->trialPeriodDays ?? 0);
        // $requestSubscriptionPricingPlan->setRecurrenceCount($request->recurrenceCount ?? 0);

        $subscriptionPricingPlan = SubscriptionPricingPlan::create($requestSubscriptionPricingPlan, $this->config);

        return $subscriptionPricingPlan;
    }

    // retrieve a subscription pricing plan for \Iyzipay\Request\Subscription\SubscriptionRetrievePricingPlanRequest
    public function retrieveSubscriptionPricingPlan($request)
    {
        $requestSubscriptionPricingPlan = new SubscriptionRetrievePricingPlanRequest;
        $requestSubscriptionPricingPlan->setLocale($this->locale);
        $requestSubscriptionPricingPlan->setConversationId($this->generateRandomNumber());
        $requestSubscriptionPricingPlan->setPricingPlanReferenceCode($request->pricingPlanReferenceCode);

        $subscriptionPricingPlan = SubscriptionPricingPlan::retrieve($requestSubscriptionPricingPlan, $this->config);

        return $subscriptionPricingPlan;
    }

    // update a subscription pricing plan for \Iyzipay\Request\Subscription\SubscriptionUpdatePricingPlanRequest
    public function updateSubscriptionPricingPlan($request)
    {
        $requestSubscriptionPricingPlan = new SubscriptionUpdatePricingPlanRequest;
        $requestSubscriptionPricingPlan->setLocale($this->locale);
        $requestSubscriptionPricingPlan->setConversationId($this->generateRandomNumber());
        $requestSubscriptionPricingPlan->setPricingPlanReferenceCode($request->pricingPlanReferenceCode);
        $requestSubscriptionPricingPlan->setName($request->name ?? '');
        $requestSubscriptionPricingPlan->setTrialPeriodDays($request->trialPeriodDays ?? '');

        $subscriptionPricingPlan = SubscriptionPricingPlan::update($requestSubscriptionPricingPlan, $this->config);

        return $subscriptionPricingPlan;
    }

    // delete a subscription pricing plan for \Iyzipay\Request\Subscription\SubscriptionDeletePricingPlanRequest
    public function deleteSubscriptionPricingPlan($request)
    {
        $requestSubscriptionPricingPlan = new SubscriptionDeletePricingPlanRequest;
        $requestSubscriptionPricingPlan->setLocale($this->locale);
        $requestSubscriptionPricingPlan->setConversationId($this->generateRandomNumber());
        $requestSubscriptionPricingPlan->setPricingPlanReferenceCode($request->pricingPlanReferenceCode);

        $subscriptionPricingPlan = SubscriptionPricingPlan::delete($requestSubscriptionPricingPlan, $this->config);

        return $subscriptionPricingPlan;
    }

    // ************************************************

    // *************** BUYER ACTIONS ******************

    // ************************************************

    // create a buyer for \Iyzipay\Model\Buyer
    public function createBuyer($request)
    {
        $buyer = new Buyer;
        $buyer->setId($request->id);
        $buyer->setName($request->name ?? '');
        $buyer->setSurname($request->surname ?? '');
        $buyer->setIdentityNumber($request->identityNumber ?? '');
        $buyer->setEmail($request->email ?? '');
        $buyer->setGsmNumber($request->gsmNumber ?? '');
        // $buyer->setRegistrationDate($request->registrationDate); //was optional so not added
        // $buyer->setLastLoginDate($request->lastLoginDate); //was optional so not added
        $buyer->setRegistrationAddress($request->registrationAddress ?? '');
        $buyer->setCity($request->city ?? '');
        $buyer->setCountry($request->country ?? '');
        $buyer->setZipCode($request->zipCode ?? '');
        $buyer->setIp($request->ip ?? '');

        return $buyer;
    }

    // ************************************************

    // ************ BASKET ITEM ACTIONS ******************

    // ************************************************

    // create a basket item for \Iyzipay\Model\BasketItem
    public function createBasketItem($request)
    {
        $basketItem = new BasketItem;
        $basketItem->setId($request['basketItemId']);
        $basketItem->setName($request['name']);
        $basketItem->setCategory1($request['category1'] ?? '');
        // $basketItem->setCategory2($request['category2'] ?? "");
        $basketItem->setItemType($request['itemType'] ?? '');
        $basketItem->setPrice($request['price'] ?? '');

        return $basketItem;
    }

    // ************************************************

    // ************ ADDRESS ACTIONS ******************

    // ************************************************

    // create a address for \Iyzipay\Model\Address
    public function createAddress($request)
    {
        $address = new Address;
        $address->setContactName($request->contactName ?? '');
        $address->setCity($request->city ?? '');
        $address->setCountry($request->country ?? '');
        $address->setAddress($request->address ?? '');
        $address->setZipCode($request->zipCode ?? '');

        return $address;
    }

    // ************************************************

    // ************ ONE TIME PAYMENT ACTIONS ******************

    // ************************************************

    // create a one time payment with \Iyzipay\Request\CreateCheckoutFormInitializeRequest
    public function createOneTimePayment($request)
    {
        $requestOneTimePayment = new CreateCheckoutFormInitializeRequest;
        $requestOneTimePayment->setLocale($this->locale);
        $requestOneTimePayment->setConversationId($this->generateRandomNumber());
        $requestOneTimePayment->setPrice($request->price ?? '');
        $requestOneTimePayment->setPaidPrice($request->paidPrice ?? '');
        $requestOneTimePayment->setCurrency($this->currency ?? '');
        $requestOneTimePayment->setBasketId($request->basketId ?? '');
        // $requestOneTimePayment->setPaymentGroup($request->paymentGroup ?? ""); // Optional so not used
        // $requestOneTimePayment->setPaymentSource($request->paymentSource ?? ""); // Optional so not used
        $requestOneTimePayment->setCallbackUrl($request->callbackUrl ?? '');
        $requestOneTimePayment->setEnabledInstallments($request->enabledInstallments ?? '');
        // $requestOneTimePayment->setDebitCardAllowed($request->debitCardAllowed ?? ""); // not in example page
        $requestOneTimePayment->setBuyer($request->buyer ?? '');
        $requestOneTimePayment->setShippingAddress($request->shippingAddress ?? '');
        $requestOneTimePayment->setBillingAddress($request->billingAddress ?? '');
        $requestOneTimePayment->setBasketItems($request->basketItems ?? '');

        $checkoutFormInitialize = CheckoutFormInitialize::create($requestOneTimePayment, $this->config);

        return $checkoutFormInitialize;
    }

    // retrieve a one time payment result with \Iyzipay\Request\RetrieveCheckoutFormRequest
    public function retrieveOneTimePayment($request)
    {
        $requestOneTimePayment = new RetrieveCheckoutFormRequest;
        $requestOneTimePayment->setLocale($this->locale);
        $requestOneTimePayment->setConversationId($this->generateRandomNumber());
        $requestOneTimePayment->setToken($request['token']);

        $checkoutForm = CheckoutForm::retrieve($requestOneTimePayment, $this->config);

        return $checkoutForm;
    }

    // ************************************************

    // ************ SUBSCRIPTION ACTIONS ******************

    // ************************************************

    // create a subscription with \Iyzipay\Request\Subscription\SubscriptionCreateCheckoutFormRequest
    public function createSubscription($request)
    {
        $requestSubscription = new SubscriptionCreateCheckoutFormRequest;
        $requestSubscription->setLocale($this->locale);
        $requestSubscription->setConversationId($this->generateRandomNumber());
        $requestSubscription->setCallbackUrl($request->callbackUrl);
        $requestSubscription->setPricingPlanReferenceCode($request->pricingPlanReferenceCode);
        $requestSubscription->setSubscriptionInitialStatus($request->subscriptionInitialStatus);
        $requestSubscription->setCustomer($request->customer);

        $checkoutFormInitialize = SubscriptionCheckoutForm::create($requestSubscription, $this->config);

        return $checkoutFormInitialize;
    }

    // get details of a subscription with \Iyzipay\Request\Subscription\SubscriptionDetailsRequest
    public function getSubscriptionDetails($request)
    {
        $requestSubscriptionDetails = new SubscriptionDetailsRequest;
        $requestSubscriptionDetails->setLocale($this->locale);
        $requestSubscriptionDetails->setConversationId($this->generateRandomNumber());
        $requestSubscriptionDetails->setSubscriptionReferenceCode($request->subscriptionReferenceCode);

        $subscriptionDetails = SubscriptionDetails::retrieve($requestSubscriptionDetails, $this->config);

        return $subscriptionDetails;
    }

    // cancel a subscription with \Iyzipay\Request\Subscription\SubscriptionCancelRequest
    public function cancelSubscription($request)
    {
        $requestSubscriptionCancel = new SubscriptionCancelRequest;
        $requestSubscriptionCancel->setLocale($this->locale);
        $requestSubscriptionCancel->setConversationId($this->generateRandomNumber());
        $requestSubscriptionCancel->setSubscriptionReferenceCode($request->subscriptionReferenceCode);

        $subscriptionCancel = SubscriptionCancel::cancel($requestSubscriptionCancel, $this->config);

        return $subscriptionCancel;
    }

    // ************************************************

    // ************ WEBHOOKS ACTIONS ******************

    // ************************************************

    // TODO: Will be implemented later

}
