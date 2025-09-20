# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Refunds\CreateRefundRequestBuilder
### Namespace: [\YooKassa\Request\Refunds](../namespaces/yookassa-request-refunds.md)
---
**Summary:**

Класс, представляющий модель CreateRefundRequestBuilder.

**Description:**

Класс билдера запросов к API на создание возврата средств.

---
### Examples
Пример использования билдера

```php
try {
    $refundBuilder = \YooKassa\Request\Refunds\CreateRefundRequest::builder();
    $refundBuilder
        ->setPaymentId('24b94598-000f-5000-9000-1b68e7b15f3f')
        ->setAmount(3500.00)
        ->setCurrency(\YooKassa\Model\CurrencyCode::RUB)
        ->setDescription('Не подошел цвет')
        ->setReceiptItems([
            (new \YooKassa\Model\Receipt\ReceiptItem)
                ->setDescription('Платок Gucci Новый')
                ->setQuantity(1)
                ->setVatCode(2)
                ->setPrice(new \YooKassa\Model\Receipt\ReceiptItemAmount(3500.00))
                ->setPaymentSubject(\YooKassa\Model\Receipt\PaymentSubject::COMMODITY)
                ->setPaymentMode(\YooKassa\Model\Receipt\PaymentMode::FULL_PAYMENT)
        ])
        ->setReceiptEmail('john.doe@merchant.com')
        ->setTaxSystemCode(1)
    ;

    // Создаем объект запроса
    $request = $refundBuilder->build();

    // Можно изменить данные, если нужно
    $request->setDescription('Не подошел цвет и размер');

    $idempotenceKey = uniqid('', true);
    $response = $client->createRefund($request, $idempotenceKey);
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
| protected | [$amount](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#property_amount) |  | Сумма |
| protected | [$currentObject](../classes/YooKassa-Request-Refunds-CreateRefundRequestBuilder.md#property_currentObject) |  | Собираемый объект запроса к API. |
| protected | [$receipt](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#property_receipt) |  | Объект с информацией о чеке |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractRequestBuilder.md#method___construct) |  | Конструктор, инициализирует пустой запрос, который в будущем начнём собирать. |
| public | [addReceiptItem()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_addReceiptItem) |  | Добавляет в чек товар |
| public | [addReceiptShipping()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_addReceiptShipping) |  | Добавляет в чек доставку товара. |
| public | [addSource()](../classes/YooKassa-Request-Refunds-CreateRefundRequestBuilder.md#method_addSource) |  | Добавляет источник возврата. |
| public | [addTransfer()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_addTransfer) |  | Добавляет трансфер. |
| public | [build()](../classes/YooKassa-Request-Refunds-CreateRefundRequestBuilder.md#method_build) |  | Строит объект запроса к API. |
| public | [setAmount()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_setAmount) |  | Устанавливает сумму. |
| public | [setCurrency()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_setCurrency) |  | Устанавливает валюту в которой будет происходить подтверждение оплаты заказа. |
| public | [setDeal()](../classes/YooKassa-Request-Refunds-CreateRefundRequestBuilder.md#method_setDeal) |  | Устанавливает сделку. |
| public | [setDescription()](../classes/YooKassa-Request-Refunds-CreateRefundRequestBuilder.md#method_setDescription) |  | Устанавливает комментарий к возврату. |
| public | [setOptions()](../classes/YooKassa-Common-AbstractRequestBuilder.md#method_setOptions) |  | Устанавливает свойства запроса из массива. |
| public | [setPaymentId()](../classes/YooKassa-Request-Refunds-CreateRefundRequestBuilder.md#method_setPaymentId) |  | Устанавливает айди платежа для которого создаётся возврат |
| public | [setReceipt()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_setReceipt) |  | Устанавливает чек. |
| public | [setReceiptEmail()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_setReceiptEmail) |  | Устанавливает адрес электронной почты получателя чека. |
| public | [setReceiptItems()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_setReceiptItems) |  | Устанавливает список товаров для создания чека. |
| public | [setReceiptPhone()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_setReceiptPhone) |  | Устанавливает телефон получателя чека. |
| public | [setRefundMethodData()](../classes/YooKassa-Request-Refunds-CreateRefundRequestBuilder.md#method_setRefundMethodData) |  | Устанавливает метод возврата. |
| public | [setSources()](../classes/YooKassa-Request-Refunds-CreateRefundRequestBuilder.md#method_setSources) |  | Устанавливает источники возврата. |
| public | [setTaxSystemCode()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_setTaxSystemCode) |  | Устанавливает код системы налогообложения. |
| public | [setTransfers()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_setTransfers) |  | Устанавливает трансферы. |
| protected | [initCurrentObject()](../classes/YooKassa-Request-Refunds-CreateRefundRequestBuilder.md#method_initCurrentObject) |  | Возвращает новый объект для сборки. |

---
### Details
* File: [lib/Request/Refunds/CreateRefundRequestBuilder.php](../../lib/Request/Refunds/CreateRefundRequestBuilder.php)
* Package: YooKassa\Request
* Class Hierarchy:  
  * [\YooKassa\Common\AbstractRequestBuilder](../classes/YooKassa-Common-AbstractRequestBuilder.md)
  * [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)
  * \YooKassa\Request\Refunds\CreateRefundRequestBuilder

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
<a name="property_amount"></a>
#### protected $amount : ?\YooKassa\Model\MonetaryAmount
---
**Summary**

Сумма

**Type:** <a href="../?\YooKassa\Model\MonetaryAmount"><abbr title="?\YooKassa\Model\MonetaryAmount">MonetaryAmount</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)


<a name="property_currentObject"></a>
#### protected $currentObject : ?\YooKassa\Common\AbstractRequestInterface
---
**Summary**

Собираемый объект запроса к API.

**Type:** <a href="../?\YooKassa\Common\AbstractRequestInterface"><abbr title="?\YooKassa\Common\AbstractRequestInterface">AbstractRequestInterface</abbr></a>

**Details:**


<a name="property_receipt"></a>
#### protected $receipt : ?\YooKassa\Model\Receipt\Receipt
---
**Summary**

Объект с информацией о чеке

**Type:** <a href="../?\YooKassa\Model\Receipt\Receipt"><abbr title="?\YooKassa\Model\Receipt\Receipt">Receipt</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)



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


<a name="method_addReceiptItem" class="anchor"></a>
#### public addReceiptItem() : self

```php
public addReceiptItem(string $title, string $price, float $quantity, int $vatCode, null|string $paymentMode = null, null|string $paymentSubject = null, null|mixed $productCode = null, null|mixed $countryOfOriginCode = null, null|mixed $customsDeclarationNumber = null, null|mixed $excise = null) : self
```

**Summary**

Добавляет в чек товар

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)
* See Also:
 * [](\YooKassa\Request\Payments\PaymentSubject)
 * [](\YooKassa\Request\Payments\PaymentMode)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | title  | Название или описание товара |
| <code lang="php">string</code> | price  | Цена товара в валюте, заданной в заказе |
| <code lang="php">float</code> | quantity  | Количество товара |
| <code lang="php">int</code> | vatCode  | Ставка НДС |
| <code lang="php">null OR string</code> | paymentMode  | значение перечисления PaymentMode |
| <code lang="php">null OR string</code> | paymentSubject  | значение перечисления PaymentSubject |
| <code lang="php">null OR mixed</code> | productCode  |  |
| <code lang="php">null OR mixed</code> | countryOfOriginCode  |  |
| <code lang="php">null OR mixed</code> | customsDeclarationNumber  |  |
| <code lang="php">null OR mixed</code> | excise  |  |

**Returns:** self - Инстанс билдера запросов


<a name="method_addReceiptShipping" class="anchor"></a>
#### public addReceiptShipping() : self

```php
public addReceiptShipping(string $title, string $price, int $vatCode, null|string $paymentMode = null, null|string $paymentSubject = null) : self
```

**Summary**

Добавляет в чек доставку товара.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)
* See Also:
 * [](\YooKassa\Request\Payments\PaymentSubject)
 * [](\YooKassa\Request\Payments\PaymentMode)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | title  | Название доставки в чеке |
| <code lang="php">string</code> | price  | Стоимость доставки |
| <code lang="php">int</code> | vatCode  | Ставка НДС |
| <code lang="php">null OR string</code> | paymentMode  | значение перечисления PaymentMode |
| <code lang="php">null OR string</code> | paymentSubject  | значение перечисления PaymentSubject |

**Returns:** self - Инстанс билдера запросов


<a name="method_addSource" class="anchor"></a>
#### public addSource() : self

```php
public addSource(array|\YooKassa\Model\Refund\SourceInterface $value) : self
```

**Summary**

Добавляет источник возврата.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestBuilder](../classes/YooKassa-Request-Refunds-CreateRefundRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Refund\SourceInterface</code> | value  | Источник возврата |

**Returns:** self - Инстанс билдера запросов


<a name="method_addTransfer" class="anchor"></a>
#### public addTransfer() : self

```php
public addTransfer(array|\YooKassa\Request\Payments\TransferDataInterface $value) : self
```

**Summary**

Добавляет трансфер.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Request\Payments\TransferDataInterface</code> | value  | Трансфер |

**Returns:** self - Инстанс билдера запросов


<a name="method_build" class="anchor"></a>
#### public build() : \YooKassa\Request\Refunds\CreateRefundRequest|\YooKassa\Common\AbstractRequestInterface

```php
public build(null|array $options = null) : \YooKassa\Request\Refunds\CreateRefundRequest|\YooKassa\Common\AbstractRequestInterface
```

**Summary**

Строит объект запроса к API.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestBuilder](../classes/YooKassa-Request-Refunds-CreateRefundRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array</code> | options  | Устанавливаемые параметры запроса |

**Returns:** \YooKassa\Request\Refunds\CreateRefundRequest|\YooKassa\Common\AbstractRequestInterface - Инстанс сгенерированного объекта запроса к API


<a name="method_setAmount" class="anchor"></a>
#### public setAmount() : self

```php
public setAmount(\YooKassa\Model\AmountInterface|array|\YooKassa\Request\Payments\numeric $value, string|null $currency = null) : self
```

**Summary**

Устанавливает сумму.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR \YooKassa\Request\Payments\numeric</code> | value  | Сумма оплаты |
| <code lang="php">string OR null</code> | currency  | Валюта |

**Returns:** self - Инстанс билдера запросов


<a name="method_setCurrency" class="anchor"></a>
#### public setCurrency() : self

```php
public setCurrency(string $value) : self
```

**Summary**

Устанавливает валюту в которой будет происходить подтверждение оплаты заказа.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | value  | Валюта в которой подтверждается оплата |

**Returns:** self - Инстанс билдера запросов


<a name="method_setDeal" class="anchor"></a>
#### public setDeal() : \YooKassa\Request\Refunds\CreateRefundRequestBuilder

```php
public setDeal(null|array|\YooKassa\Model\Deal\RefundDealData $value) : \YooKassa\Request\Refunds\CreateRefundRequestBuilder
```

**Summary**

Устанавливает сделку.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestBuilder](../classes/YooKassa-Request-Refunds-CreateRefundRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Model\Deal\RefundDealData</code> | value  | Данные о сделке, в составе которой проходит возврат |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException |  |

**Returns:** \YooKassa\Request\Refunds\CreateRefundRequestBuilder - Инстанс билдера запросов


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : \YooKassa\Request\Refunds\CreateRefundRequestBuilder

```php
public setDescription(string|null $value) : \YooKassa\Request\Refunds\CreateRefundRequestBuilder
```

**Summary**

Устанавливает комментарий к возврату.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestBuilder](../classes/YooKassa-Request-Refunds-CreateRefundRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Комментарий к возврату |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException | Выбрасывается если была передана не строка |

**Returns:** \YooKassa\Request\Refunds\CreateRefundRequestBuilder - Инстанс текущего билдера


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


<a name="method_setPaymentId" class="anchor"></a>
#### public setPaymentId() : \YooKassa\Request\Refunds\CreateRefundRequestBuilder

```php
public setPaymentId(string|null $value) : \YooKassa\Request\Refunds\CreateRefundRequestBuilder
```

**Summary**

Устанавливает айди платежа для которого создаётся возврат

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestBuilder](../classes/YooKassa-Request-Refunds-CreateRefundRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Айди платежа |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\EmptyPropertyValueException | Выбрасывается если передано пустое значение айди платежа |
| \YooKassa\Common\Exceptions\InvalidPropertyValueException | Выбрасывается если переданное значение является строкой, но не является валидным значением айди платежа |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException | Выбрасывается если передано значение не валидного типа |

**Returns:** \YooKassa\Request\Refunds\CreateRefundRequestBuilder - Инстанс текущего билдера


<a name="method_setReceipt" class="anchor"></a>
#### public setReceipt() : self

```php
public setReceipt(array|\YooKassa\Model\Receipt\ReceiptInterface $value) : self
```

**Summary**

Устанавливает чек.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\ReceiptInterface</code> | value  | Инстанс чека или ассоциативный массив с данными чека |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException | Генерируется если было передано значение невалидного типа |

**Returns:** self - Инстанс билдера запросов


<a name="method_setReceiptEmail" class="anchor"></a>
#### public setReceiptEmail() : self

```php
public setReceiptEmail(string|null $value) : self
```

**Summary**

Устанавливает адрес электронной почты получателя чека.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Email получателя чека |

**Returns:** self - Инстанс билдера запросов


<a name="method_setReceiptItems" class="anchor"></a>
#### public setReceiptItems() : self

```php
public setReceiptItems(array $value = []) : self
```

**Summary**

Устанавливает список товаров для создания чека.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | value  | Массив товаров в заказе |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueException | Выбрасывается если хотя бы один из товаров имеет неверную структуру |

**Returns:** self - Инстанс билдера запросов


<a name="method_setReceiptPhone" class="anchor"></a>
#### public setReceiptPhone() : self

```php
public setReceiptPhone(string|null $value) : self
```

**Summary**

Устанавливает телефон получателя чека.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Телефон получателя чека |

**Returns:** self - Инстанс билдера запросов


<a name="method_setRefundMethodData" class="anchor"></a>
#### public setRefundMethodData() : self

```php
public setRefundMethodData(\YooKassa\Request\Refunds\RefundMethodData\AbstractRefundMethodData|array|null $value = null) : self
```

**Summary**

Устанавливает метод возврата.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestBuilder](../classes/YooKassa-Request-Refunds-CreateRefundRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Refunds\RefundMethodData\AbstractRefundMethodData OR array OR null</code> | value  |  |

**Returns:** self - 


<a name="method_setSources" class="anchor"></a>
#### public setSources() : self

```php
public setSources(array|\YooKassa\Model\Refund\SourceInterface[]|\YooKassa\Common\ListObjectInterface $value) : self
```

**Summary**

Устанавливает источники возврата.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestBuilder](../classes/YooKassa-Request-Refunds-CreateRefundRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Refund\SourceInterface[] OR \YooKassa\Common\ListObjectInterface</code> | value  | Массив трансферов |

**Returns:** self - Инстанс билдера запросов


<a name="method_setTaxSystemCode" class="anchor"></a>
#### public setTaxSystemCode() : self

```php
public setTaxSystemCode(int|null $value) : self
```

**Summary**

Устанавливает код системы налогообложения.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int OR null</code> | value  | Код системы налогообложения. Число 1-6. |

**Returns:** self - Инстанс билдера запросов


<a name="method_setTransfers" class="anchor"></a>
#### public setTransfers() : self

```php
public setTransfers(array|\YooKassa\Request\Payments\TransferDataInterface[]|\YooKassa\Common\ListObjectInterface $value) : self
```

**Summary**

Устанавливает трансферы.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Request\Payments\TransferDataInterface[] OR \YooKassa\Common\ListObjectInterface</code> | value  | Массив трансферов |

**Returns:** self - Инстанс билдера запросов


<a name="method_initCurrentObject" class="anchor"></a>
#### protected initCurrentObject() : \YooKassa\Request\Refunds\CreateRefundRequest

```php
protected initCurrentObject() : \YooKassa\Request\Refunds\CreateRefundRequest
```

**Summary**

Возвращает новый объект для сборки.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestBuilder](../classes/YooKassa-Request-Refunds-CreateRefundRequestBuilder.md)

**Returns:** \YooKassa\Request\Refunds\CreateRefundRequest - Собираемый объект запроса к API



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