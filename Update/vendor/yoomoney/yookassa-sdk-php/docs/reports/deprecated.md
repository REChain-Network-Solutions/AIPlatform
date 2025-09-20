# [YooKassa API SDK](../home.md)

# Deprecated Elements
### Table of Contents
* [lib/Common/Exceptions/ApiException.php](../../lib/Common/Exceptions/ApiException.php)
* [lib/Model/Invoice/DeliveryMethod/DeliveryMethodType.php](../../lib/Model/Invoice/DeliveryMethod/DeliveryMethodType.php)
* [lib/Model/Invoice/DeliveryMethod/DeliveryMethodUnknown.php](../../lib/Model/Invoice/DeliveryMethod/DeliveryMethodUnknown.php)
* [lib/Model/Payment/CancellationDetailsPartyCode.php](../../lib/Model/Payment/CancellationDetailsPartyCode.php)
* [lib/Model/Payment/ConfirmationType.php](../../lib/Model/Payment/ConfirmationType.php)
* [lib/Model/Payment/PaymentMethod/B2b/Sberbank/VatDataRate.php](../../lib/Model/Payment/PaymentMethod/B2b/Sberbank/VatDataRate.php)
* [lib/Model/Payment/PaymentMethod/PaymentMethodAlfaBank.php](../../lib/Model/Payment/PaymentMethod/PaymentMethodAlfaBank.php)
* [lib/Model/Payment/PaymentMethod/PaymentMethodFactory.php](../../lib/Model/Payment/PaymentMethod/PaymentMethodFactory.php)
* [lib/Model/Payment/PaymentMethod/PaymentMethodInstallments.php](../../lib/Model/Payment/PaymentMethod/PaymentMethodInstallments.php)
* [lib/Model/Payment/PaymentMethod/PaymentMethodPsb.php](../../lib/Model/Payment/PaymentMethod/PaymentMethodPsb.php)
* [lib/Model/Payment/PaymentMethod/PaymentMethodQiwi.php](../../lib/Model/Payment/PaymentMethod/PaymentMethodQiwi.php)
* [lib/Model/Payment/PaymentMethod/PaymentMethodUnknown.php](../../lib/Model/Payment/PaymentMethod/PaymentMethodUnknown.php)
* [lib/Model/Payment/PaymentMethod/PaymentMethodWebmoney.php](../../lib/Model/Payment/PaymentMethod/PaymentMethodWebmoney.php)
* [lib/Model/Payment/PaymentMethod/PaymentMethodWechat.php](../../lib/Model/Payment/PaymentMethod/PaymentMethodWechat.php)
* [lib/Model/Payment/PaymentMethodType.php](../../lib/Model/Payment/PaymentMethodType.php)
* [lib/Model/Refund/RefundCancellationDetailsPartyCode.php](../../lib/Model/Refund/RefundCancellationDetailsPartyCode.php)
* [lib/Model/Refund/RefundMethod/RefundMethodUnknown.php](../../lib/Model/Refund/RefundMethod/RefundMethodUnknown.php)
* [lib/Model/Refund/RefundMethodType.php](../../lib/Model/Refund/RefundMethodType.php)
* [lib/Model/Settings/Me.php](../../lib/Model/Settings/Me.php)
* [lib/Request/Payments/CreatePaymentRequest.php](../../lib/Request/Payments/CreatePaymentRequest.php)
* [lib/Request/Payments/CreatePaymentRequestBuilder.php](../../lib/Request/Payments/CreatePaymentRequestBuilder.php)
* [lib/Request/Payments/FraudData.php](../../lib/Request/Payments/FraudData.php)
* [lib/Request/Payments/PaymentData/PaymentDataAlfabank.php](../../lib/Request/Payments/PaymentData/PaymentDataAlfabank.php)
* [lib/Request/Payments/PaymentData/PaymentDataInstallments.php](../../lib/Request/Payments/PaymentData/PaymentDataInstallments.php)
* [lib/Request/Payments/PaymentData/PaymentDataQiwi.php](../../lib/Request/Payments/PaymentData/PaymentDataQiwi.php)

<a id="lib/Common/Exceptions/ApiException.php"></a>
#### [lib/Common/Exceptions/ApiException.php](../../lib/Common/Exceptions/ApiException.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 154 | \YooKassa\Common\Exceptions\ApiException::getErrorId() | Устарело. Вместо него нужно использовать getError()-&gt;getId() |
| 161 | \YooKassa\Common\Exceptions\ApiException::getErrorCode() | Устарело. Вместо него нужно использовать getError()-&gt;getCode() |
| 168 | \YooKassa\Common\Exceptions\ApiException::getErrorDescription() | Устарело. Вместо него нужно использовать getError()-&gt;getDescription() |
| 175 | \YooKassa\Common\Exceptions\ApiException::getErrorParameter() | Устарело. Вместо него нужно использовать getError()-&gt;getParameter() |
| 57 | \YooKassa\Common\Exceptions\ApiException::$type | Устарело. Вместо него нужно использовать getError()-&gt;getType(). |
| 63 | \YooKassa\Common\Exceptions\ApiException::$retryAfter | Устарело. Вместо него нужно использовать getError()-&gt;getRetryAfter(). |
<a id="lib/Model/Invoice/DeliveryMethod/DeliveryMethodType.php"></a>
#### [lib/Model/Invoice/DeliveryMethod/DeliveryMethodType.php](../../lib/Model/Invoice/DeliveryMethod/DeliveryMethodType.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 55 | \YooKassa\Model\Invoice\DeliveryMethod\DeliveryMethodType::UNKNOWN | Не используется для реальных платежей |
<a id="lib/Model/Invoice/DeliveryMethod/DeliveryMethodUnknown.php"></a>
#### [lib/Model/Invoice/DeliveryMethod/DeliveryMethodUnknown.php](../../lib/Model/Invoice/DeliveryMethod/DeliveryMethodUnknown.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 41 | \YooKassa\Model\Invoice\DeliveryMethod\DeliveryMethodUnknown | Не используется в реальных счетах |
<a id="lib/Model/Payment/CancellationDetailsPartyCode.php"></a>
#### [lib/Model/Payment/CancellationDetailsPartyCode.php](../../lib/Model/Payment/CancellationDetailsPartyCode.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 56 | \YooKassa\Model\Payment\CancellationDetailsPartyCode::YANDEX_CHECKOUT | Устарел. Оставлен для обратной совместимости |
<a id="lib/Model/Payment/ConfirmationType.php"></a>
#### [lib/Model/Payment/ConfirmationType.php](../../lib/Model/Payment/ConfirmationType.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 62 | \YooKassa\Model\Payment\ConfirmationType::CODE_VERIFICATION | Будет удален в следующих версиях |
<a id="lib/Model/Payment/PaymentMethod/B2b/Sberbank/VatDataRate.php"></a>
#### [lib/Model/Payment/PaymentMethod/B2b/Sberbank/VatDataRate.php](../../lib/Model/Payment/PaymentMethod/B2b/Sberbank/VatDataRate.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 62 | \YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\VatDataRate::RATE_18 | Больше не применяется |
<a id="lib/Model/Payment/PaymentMethod/PaymentMethodAlfaBank.php"></a>
#### [lib/Model/Payment/PaymentMethod/PaymentMethodAlfaBank.php](../../lib/Model/Payment/PaymentMethod/PaymentMethodAlfaBank.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 45 | \YooKassa\Model\Payment\PaymentMethod\PaymentMethodAlfaBank | Будет удален в следующих версиях |
<a id="lib/Model/Payment/PaymentMethod/PaymentMethodFactory.php"></a>
#### [lib/Model/Payment/PaymentMethod/PaymentMethodFactory.php](../../lib/Model/Payment/PaymentMethod/PaymentMethodFactory.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 45 | \YooKassa\Model\Payment\PaymentMethod\PaymentMethodFactory::YANDEX_MONEY | Для поддержки старых платежей |
<a id="lib/Model/Payment/PaymentMethod/PaymentMethodInstallments.php"></a>
#### [lib/Model/Payment/PaymentMethod/PaymentMethodInstallments.php](../../lib/Model/Payment/PaymentMethod/PaymentMethodInstallments.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 42 | \YooKassa\Model\Payment\PaymentMethod\PaymentMethodInstallments | Будет удален в следующих версиях |
<a id="lib/Model/Payment/PaymentMethod/PaymentMethodPsb.php"></a>
#### [lib/Model/Payment/PaymentMethod/PaymentMethodPsb.php](../../lib/Model/Payment/PaymentMethod/PaymentMethodPsb.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 42 | \YooKassa\Model\Payment\PaymentMethod\PaymentMethodPsb | Будет удален в следующих версиях |
<a id="lib/Model/Payment/PaymentMethod/PaymentMethodQiwi.php"></a>
#### [lib/Model/Payment/PaymentMethod/PaymentMethodQiwi.php](../../lib/Model/Payment/PaymentMethod/PaymentMethodQiwi.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 42 | \YooKassa\Model\Payment\PaymentMethod\PaymentMethodQiwi | Будет удален в следующих версиях |
<a id="lib/Model/Payment/PaymentMethod/PaymentMethodUnknown.php"></a>
#### [lib/Model/Payment/PaymentMethod/PaymentMethodUnknown.php](../../lib/Model/Payment/PaymentMethod/PaymentMethodUnknown.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 42 | \YooKassa\Model\Payment\PaymentMethod\PaymentMethodUnknown | Не используется в реальных платежах |
<a id="lib/Model/Payment/PaymentMethod/PaymentMethodWebmoney.php"></a>
#### [lib/Model/Payment/PaymentMethod/PaymentMethodWebmoney.php](../../lib/Model/Payment/PaymentMethod/PaymentMethodWebmoney.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 42 | \YooKassa\Model\Payment\PaymentMethod\PaymentMethodWebmoney | Будет удален в следующих версиях |
<a id="lib/Model/Payment/PaymentMethod/PaymentMethodWechat.php"></a>
#### [lib/Model/Payment/PaymentMethod/PaymentMethodWechat.php](../../lib/Model/Payment/PaymentMethod/PaymentMethodWechat.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 42 | \YooKassa\Model\Payment\PaymentMethod\PaymentMethodWechat | Будет удален в следующих версиях |
<a id="lib/Model/Payment/PaymentMethodType.php"></a>
#### [lib/Model/Payment/PaymentMethodType.php](../../lib/Model/Payment/PaymentMethodType.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 87 | \YooKassa\Model\Payment\PaymentMethodType::QIWI | Больше недоступен |
| 94 | \YooKassa\Model\Payment\PaymentMethodType::WEBMONEY | Больше недоступен |
| 101 | \YooKassa\Model\Payment\PaymentMethodType::ALFABANK | Больше недоступен |
| 114 | \YooKassa\Model\Payment\PaymentMethodType::PSB | Больше недоступен |
| 121 | \YooKassa\Model\Payment\PaymentMethodType::INSTALLMENTS | Больше недоступен |
| 128 | \YooKassa\Model\Payment\PaymentMethodType::WECHAT | Больше недоступен |
| 147 | \YooKassa\Model\Payment\PaymentMethodType::UNKNOWN | Не используется для реальных платежей |
<a id="lib/Model/Refund/RefundCancellationDetailsPartyCode.php"></a>
#### [lib/Model/Refund/RefundCancellationDetailsPartyCode.php](../../lib/Model/Refund/RefundCancellationDetailsPartyCode.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 51 | \YooKassa\Model\Refund\RefundCancellationDetailsPartyCode::YANDEX_CHECKOUT | Устарел. Оставлен для обратной совместимости |
<a id="lib/Model/Refund/RefundMethod/RefundMethodUnknown.php"></a>
#### [lib/Model/Refund/RefundMethod/RefundMethodUnknown.php](../../lib/Model/Refund/RefundMethod/RefundMethodUnknown.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 42 | \YooKassa\Model\Refund\RefundMethod\RefundMethodUnknown | Не используется в реальных возвратах |
<a id="lib/Model/Refund/RefundMethodType.php"></a>
#### [lib/Model/Refund/RefundMethodType.php](../../lib/Model/Refund/RefundMethodType.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 56 | \YooKassa\Model\Refund\RefundMethodType::UNKNOWN | Не используется для реальных возвратов |
<a id="lib/Model/Settings/Me.php"></a>
#### [lib/Model/Settings/Me.php](../../lib/Model/Settings/Me.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 252 | \YooKassa\Model\Settings\Me::getFiscalizationEnabled() | Устарел. |
| 265 | \YooKassa\Model\Settings\Me::setFiscalizationEnabled() | Устарел. |
<a id="lib/Request/Payments/CreatePaymentRequest.php"></a>
#### [lib/Request/Payments/CreatePaymentRequest.php](../../lib/Request/Payments/CreatePaymentRequest.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 570 | \YooKassa\Request\Payments\CreatePaymentRequest::getFraudData() | Больше не поддерживается. Вместо него нужно использовать `getReceiver()` |
| 581 | \YooKassa\Request\Payments\CreatePaymentRequest::setFraudData() | Больше не поддерживается. Вместо него нужно использовать `setReceiver()` |
| 592 | \YooKassa\Request\Payments\CreatePaymentRequest::hasFraudData() | Больше не поддерживается. Вместо него нужно использовать `hasReceiver()` |
<a id="lib/Request/Payments/CreatePaymentRequestBuilder.php"></a>
#### [lib/Request/Payments/CreatePaymentRequestBuilder.php](../../lib/Request/Payments/CreatePaymentRequestBuilder.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 93 | \YooKassa\Request\Payments\CreatePaymentRequestBuilder::setAccountId() | Больше не используется |
| 330 | \YooKassa\Request\Payments\CreatePaymentRequestBuilder::setFraudData() | Больше не поддерживается. Вместо него нужно использовать `setReceiver()` |
<a id="lib/Request/Payments/FraudData.php"></a>
#### [lib/Request/Payments/FraudData.php](../../lib/Request/Payments/FraudData.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 49 | \YooKassa\Request\Payments\FraudData | Больше не поддерживается. Вместо него нужно использовать `receiver`. |
<a id="lib/Request/Payments/PaymentData/PaymentDataAlfabank.php"></a>
#### [lib/Request/Payments/PaymentData/PaymentDataAlfabank.php](../../lib/Request/Payments/PaymentData/PaymentDataAlfabank.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 46 | \YooKassa\Request\Payments\PaymentData\PaymentDataAlfabank | Будет удален в следующих версиях |
<a id="lib/Request/Payments/PaymentData/PaymentDataInstallments.php"></a>
#### [lib/Request/Payments/PaymentData/PaymentDataInstallments.php](../../lib/Request/Payments/PaymentData/PaymentDataInstallments.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 42 | \YooKassa\Request\Payments\PaymentData\PaymentDataInstallments | Будет удален в следующих версиях |
<a id="lib/Request/Payments/PaymentData/PaymentDataQiwi.php"></a>
#### [lib/Request/Payments/PaymentData/PaymentDataQiwi.php](../../lib/Request/Payments/PaymentData/PaymentDataQiwi.php)
| Line | Element | Description |
| ---- | ------- | ----------- |
| 45 | \YooKassa\Request\Payments\PaymentData\PaymentDataQiwi | Будет удален в следующих версиях |

---

### Top Namespaces

* [\YooKassa](../namespaces/yookassa.md)

---

### Reports
* [Errors - 0](../reports/errors.md)
* [Markers - 0](../reports/markers.md)
* [Deprecated - 40](../reports/deprecated.md)

---

This document was automatically generated from source code comments on 2025-09-04 using [phpDocumentor](http://www.phpdoc.org/)

&copy; 2025 YooMoney