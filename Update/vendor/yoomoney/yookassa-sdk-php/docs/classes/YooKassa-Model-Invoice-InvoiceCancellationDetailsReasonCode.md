# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Invoice\InvoiceCancellationDetailsReasonCode
### Namespace: [\YooKassa\Model\Invoice](../namespaces/yookassa-model-invoice.md)
---
**Summary:**

Класс, представляющий модель InvoiceCancellationDetailsReasonCode.

**Description:**

Возможные причины отмены счета.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [INVOICE_CANCELED](../classes/YooKassa-Model-Invoice-InvoiceCancellationDetailsReasonCode.md#constant_INVOICE_CANCELED) |  | [Счет отменен вручную](https://yookassa.ru/docs/support/merchant/invoices-to-clients/invoicing#invoicing__cancel) из личного кабинета ЮKassa. |
| public | [INVOICE_EXPIRED](../classes/YooKassa-Model-Invoice-InvoiceCancellationDetailsReasonCode.md#constant_INVOICE_EXPIRED) |  | Истек срок действия счета, который вы установили в запросе на создание счета в параметре `expires_at`, и по счету нет ни одного успешного платежа |
| public | [GENERAL_DECLINE](../classes/YooKassa-Model-Invoice-InvoiceCancellationDetailsReasonCode.md#constant_GENERAL_DECLINE) |  | Причина не детализирована, поэтому пользователю следует обратиться к инициатору отмены счета за уточнением подробностей |
| public | [PAYMENT_CANCELED](../classes/YooKassa-Model-Invoice-InvoiceCancellationDetailsReasonCode.md#constant_PAYMENT_CANCELED) |  | [Платеж отменен по API](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#cancel) при оплате в две стадии |
| public | [PAYMENT_EXPIRED_ON_CAPTURE](../classes/YooKassa-Model-Invoice-InvoiceCancellationDetailsReasonCode.md#constant_PAYMENT_EXPIRED_ON_CAPTURE) |  | [Истек срок списания оплаты](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#hold) для платежа в две стадии |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$validValues](../classes/YooKassa-Model-Invoice-InvoiceCancellationDetailsReasonCode.md#property_validValues) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEnabledValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getEnabledValues) |  | Возвращает значения в enum'е значения которых разрешены. |
| public | [getValidValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getValidValues) |  | Возвращает все значения в enum'e. |
| public | [valueExists()](../classes/YooKassa-Common-AbstractEnum.md#method_valueExists) |  | Проверяет наличие значения в enum'e. |

---
### Details
* File: [lib/Model/Invoice/InvoiceCancellationDetailsReasonCode.php](../../lib/Model/Invoice/InvoiceCancellationDetailsReasonCode.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)
  * \YooKassa\Model\Invoice\InvoiceCancellationDetailsReasonCode

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Constants
<a name="constant_INVOICE_CANCELED" class="anchor"></a>
###### INVOICE_CANCELED
[Счет отменен вручную](https://yookassa.ru/docs/support/merchant/invoices-to-clients/invoicing#invoicing__cancel) из личного кабинета ЮKassa.

```php
INVOICE_CANCELED = 'invoice_canceled'
```


<a name="constant_INVOICE_EXPIRED" class="anchor"></a>
###### INVOICE_EXPIRED
Истек срок действия счета, который вы установили в запросе на создание счета в параметре `expires_at`, и по счету нет ни одного успешного платежа

```php
INVOICE_EXPIRED = 'invoice_expired'
```


<a name="constant_GENERAL_DECLINE" class="anchor"></a>
###### GENERAL_DECLINE
Причина не детализирована, поэтому пользователю следует обратиться к инициатору отмены счета за уточнением подробностей

```php
GENERAL_DECLINE = 'general_decline'
```


<a name="constant_PAYMENT_CANCELED" class="anchor"></a>
###### PAYMENT_CANCELED
[Платеж отменен по API](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#cancel) при оплате в две стадии

```php
PAYMENT_CANCELED = 'payment_canceled'
```


<a name="constant_PAYMENT_EXPIRED_ON_CAPTURE" class="anchor"></a>
###### PAYMENT_EXPIRED_ON_CAPTURE
[Истек срок списания оплаты](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#hold) для платежа в две стадии

```php
PAYMENT_EXPIRED_ON_CAPTURE = 'payment_expired_on_capture'
```



---
## Properties
<a name="property_validValues"></a>
#### protected $validValues : array
---
**Type:** <a href="../array"><abbr title="array">array</abbr></a>
Массив принимаемых enum&#039;ом значений
**Details:**



---
## Methods
<a name="method_getEnabledValues" class="anchor"></a>
#### public getEnabledValues() : string[]

```php
Static public getEnabledValues() : string[]
```

**Summary**

Возвращает значения в enum'е значения которых разрешены.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)

**Returns:** string[] - Массив разрешённых значений


<a name="method_getValidValues" class="anchor"></a>
#### public getValidValues() : array

```php
Static public getValidValues() : array
```

**Summary**

Возвращает все значения в enum'e.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)

**Returns:** array - Массив значений в перечислении


<a name="method_valueExists" class="anchor"></a>
#### public valueExists() : bool

```php
Static public valueExists(mixed $value) : bool
```

**Summary**

Проверяет наличие значения в enum'e.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">mixed</code> | value  | Проверяемое значение |

**Returns:** bool - True если значение имеется, false если нет



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