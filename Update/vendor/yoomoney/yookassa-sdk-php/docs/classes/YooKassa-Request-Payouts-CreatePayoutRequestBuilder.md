# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payouts\CreatePayoutRequestBuilder
### Namespace: [\YooKassa\Request\Payouts](../namespaces/yookassa-request-payouts.md)
---
**Summary:**

Класс, представляющий модель CreatePayoutRequestBuilder.

**Description:**

Класс билдера объектов запросов к API на создание платежа.

---
### Examples
Пример использования билдера

```php
try {
    $payoutBuilder = \YooKassa\Request\Payouts\CreatePayoutRequest::builder();
    $payoutBuilder
        ->setAmount(new \YooKassa\Model\MonetaryAmount(80))
        ->setPayoutDestinationData(
            new \YooKassa\Request\Payouts\PayoutDestinationData\PayoutDestinationDataYooMoney(
                [
                    'type' => \YooKassa\Model\Payment\PaymentMethodType::YOO_MONEY,
                    'account_number' => '4100116075156746'
                ]
            )
        )
        ->setDeal(new \YooKassa\Model\Deal\PayoutDealInfo(['id' => 'dl-2909e77d-0022-5000-8000-0c37205b3208']))
        ->setDescription('Выплата по заказу №37')
    ;

    // Создаем объект запроса
    $request = $payoutBuilder->build();

    $idempotenceKey = uniqid('', true);
    $response = $client->createPayout($request, $idempotenceKey);
} catch (Exception $e) {
    $response = $e;
}

var_dump($response);

```

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$currentObject](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md#property_currentObject) |  | Собираемый объект запроса. |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractRequestBuilder.md#method___construct) |  | Конструктор, инициализирует пустой запрос, который в будущем начнём собирать. |
| public | [build()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md#method_build) |  | Строит и возвращает объект запроса для отправки в API ЮKassa. |
| public | [setAmount()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md#method_setAmount) |  | Устанавливает сумму. |
| public | [setDeal()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md#method_setDeal) |  | Устанавливает сделку, в рамках которой нужно провести выплату. |
| public | [setDescription()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md#method_setDescription) |  | Устанавливает описание транзакции. |
| public | [setMetadata()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md#method_setMetadata) |  | Устанавливает метаданные, привязанные к платежу. |
| public | [setOptions()](../classes/YooKassa-Common-AbstractRequestBuilder.md#method_setOptions) |  | Устанавливает свойства запроса из массива. |
| public | [setPaymentMethodId()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md#method_setPaymentMethodId) |  | Устанавливает идентификатор сохраненного способа оплаты. |
| public | [setPayoutDestinationData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md#method_setPayoutDestinationData) |  | Устанавливает объект с информацией для создания метода оплаты. |
| public | [setPayoutToken()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md#method_setPayoutToken) |  | Устанавливает одноразовый токен для проведения выплаты. |
| public | [setPersonalData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md#method_setPersonalData) |  | Устанавливает персональные данные получателя выплаты. |
| public | [setReceiptData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md#method_setReceiptData) |  | Устанавливает данные для формирования чека в сервисе Мой налог. |
| public | [setSelfEmployed()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md#method_setSelfEmployed) |  | Устанавливает данные самозанятого, который получит выплату. |
| protected | [initCurrentObject()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md#method_initCurrentObject) |  | Инициализирует объект запроса, который в дальнейшем будет собираться билдером |

---
### Details
* File: [lib/Request/Payouts/CreatePayoutRequestBuilder.php](../../lib/Request/Payouts/CreatePayoutRequestBuilder.php)
* Package: YooKassa\Request
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractRequestBuilder](../classes/YooKassa-Common-AbstractRequestBuilder.md)
  * \YooKassa\Request\Payouts\CreatePayoutRequestBuilder

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Properties
<a name="property_currentObject"></a>
#### protected $currentObject : ?\YooKassa\Common\AbstractRequestInterface
---
**Summary**

Собираемый объект запроса.

**Type:** <a href="../?\YooKassa\Common\AbstractRequestInterface"><abbr title="?\YooKassa\Common\AbstractRequestInterface">AbstractRequestInterface</abbr></a>

**Details:**



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct() : mixed
```

**Summary**

Конструктор, инициализирует пустой запрос, который в будущем начнём собирать.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequestBuilder](../classes/YooKassa-Common-AbstractRequestBuilder.md)

**Returns:** mixed - 


<a name="method_build" class="anchor"></a>
#### public build() : \YooKassa\Request\Payouts\CreatePayoutRequestInterface|\YooKassa\Common\AbstractRequestInterface

```php
public build(null|array $options = null) : \YooKassa\Request\Payouts\CreatePayoutRequestInterface|\YooKassa\Common\AbstractRequestInterface
```

**Summary**

Строит и возвращает объект запроса для отправки в API ЮKassa.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestBuilder](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array</code> | options  | Массив параметров для установки в объект запроса |

**Returns:** \YooKassa\Request\Payouts\CreatePayoutRequestInterface|\YooKassa\Common\AbstractRequestInterface - Инстанс объекта запроса


<a name="method_setAmount" class="anchor"></a>
#### public setAmount() : self

```php
public setAmount(\YooKassa\Model\AmountInterface|array|string $value) : self
```

**Summary**

Устанавливает сумму.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestBuilder](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR string</code> | value  | Сумма выплаты |

**Returns:** self - Инстанс билдера запросов


<a name="method_setDeal" class="anchor"></a>
#### public setDeal() : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder

```php
public setDeal(array|\YooKassa\Model\Deal\PayoutDealInfo $value) : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder
```

**Summary**

Устанавливает сделку, в рамках которой нужно провести выплату.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestBuilder](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Deal\PayoutDealInfo</code> | value  | Сделка, в рамках которой нужно провести выплату |

**Returns:** \YooKassa\Request\Payouts\CreatePayoutRequestBuilder - 


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder

```php
public setDescription(string|null $value) : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder
```

**Summary**

Устанавливает описание транзакции.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestBuilder](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Описание транзакции |

**Returns:** \YooKassa\Request\Payouts\CreatePayoutRequestBuilder - Инстанс текущего билдера


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder

```php
public setMetadata(null|array|\YooKassa\Model\Metadata $value) : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder
```

**Summary**

Устанавливает метаданные, привязанные к платежу.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestBuilder](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Model\Metadata</code> | value  | Метаданные платежа, устанавливаемые мерчантом |

**Returns:** \YooKassa\Request\Payouts\CreatePayoutRequestBuilder - Инстанс текущего билдера


<a name="method_setOptions" class="anchor"></a>
#### public setOptions() : \YooKassa\Common\AbstractRequestBuilder

```php
public setOptions(iterable|null $options) : \YooKassa\Common\AbstractRequestBuilder
```

**Summary**

Устанавливает свойства запроса из массива.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequestBuilder](../classes/YooKassa-Common-AbstractRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">iterable OR null</code> | options  | Массив свойств запроса |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \InvalidArgumentException | Выбрасывается если аргумент не массив и не итерируемый объект |
| \YooKassa\Common\Exceptions\InvalidPropertyException | Выбрасывается если не удалось установить один из параметров, переданных в массиве настроек |

**Returns:** \YooKassa\Common\AbstractRequestBuilder - Инстанс текущего билдера запросов


<a name="method_setPaymentMethodId" class="anchor"></a>
#### public setPaymentMethodId() : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder

```php
public setPaymentMethodId(null|string $value) : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder
```

**Summary**

Устанавливает идентификатор сохраненного способа оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestBuilder](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | value  | Идентификатор сохраненного способа оплаты |

**Returns:** \YooKassa\Request\Payouts\CreatePayoutRequestBuilder - 


<a name="method_setPayoutDestinationData" class="anchor"></a>
#### public setPayoutDestinationData() : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder

```php
public setPayoutDestinationData(null|\YooKassa\Model\Payout\AbstractPayoutDestination|array $value) : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder
```

**Summary**

Устанавливает объект с информацией для создания метода оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestBuilder](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \YooKassa\Model\Payout\AbstractPayoutDestination OR array</code> | value  | Объект создания метода оплаты или null |

**Returns:** \YooKassa\Request\Payouts\CreatePayoutRequestBuilder - 


<a name="method_setPayoutToken" class="anchor"></a>
#### public setPayoutToken() : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder

```php
public setPayoutToken(string|null $value) : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder
```

**Summary**

Устанавливает одноразовый токен для проведения выплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestBuilder](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Одноразовый токен для проведения выплаты |

**Returns:** \YooKassa\Request\Payouts\CreatePayoutRequestBuilder - Инстанс текущего билдера


<a name="method_setPersonalData" class="anchor"></a>
#### public setPersonalData() : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder

```php
public setPersonalData(null|array|\YooKassa\Request\Payouts\PayoutPersonalData[] $value) : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder
```

**Summary**

Устанавливает персональные данные получателя выплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestBuilder](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Request\Payouts\PayoutPersonalData[]</code> | value  | Персональные данные получателя выплаты |

**Returns:** \YooKassa\Request\Payouts\CreatePayoutRequestBuilder - 


<a name="method_setReceiptData" class="anchor"></a>
#### public setReceiptData() : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder

```php
public setReceiptData(null|array|\YooKassa\Request\Payouts\IncomeReceiptData $value) : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder
```

**Summary**

Устанавливает данные для формирования чека в сервисе Мой налог.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestBuilder](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Request\Payouts\IncomeReceiptData</code> | value  | Данные для формирования чека в сервисе Мой налог |

**Returns:** \YooKassa\Request\Payouts\CreatePayoutRequestBuilder - 


<a name="method_setSelfEmployed" class="anchor"></a>
#### public setSelfEmployed() : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder

```php
public setSelfEmployed(null|array|\YooKassa\Request\Payouts\PayoutSelfEmployedInfo $value) : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder
```

**Summary**

Устанавливает данные самозанятого, который получит выплату.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestBuilder](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Request\Payouts\PayoutSelfEmployedInfo</code> | value  | Данные самозанятого, который получит выплату |

**Returns:** \YooKassa\Request\Payouts\CreatePayoutRequestBuilder - 


<a name="method_initCurrentObject" class="anchor"></a>
#### protected initCurrentObject() : \YooKassa\Request\Payouts\CreatePayoutRequest

```php
protected initCurrentObject() : \YooKassa\Request\Payouts\CreatePayoutRequest
```

**Summary**

Инициализирует объект запроса, который в дальнейшем будет собираться билдером

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestBuilder](../classes/YooKassa-Request-Payouts-CreatePayoutRequestBuilder.md)

**Returns:** \YooKassa\Request\Payouts\CreatePayoutRequest - Инстанс собираемого объекта запроса к API



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