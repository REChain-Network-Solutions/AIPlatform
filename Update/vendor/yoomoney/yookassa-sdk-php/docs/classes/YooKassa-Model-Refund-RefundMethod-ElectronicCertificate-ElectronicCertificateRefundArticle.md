# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Refund\RefundMethod\ElectronicCertificate\ElectronicCertificateRefundArticle
### Namespace: [\YooKassa\Model\Refund\RefundMethod\ElectronicCertificate](../namespaces/yookassa-model-refund-refundmethod-electroniccertificate.md)
---
**Summary:**

Класс, представляющий модель ElectronicCertificateRefundArticle.

**Description:**

Товарная позиция в корзине возврата при возврате на электронный сертификат.

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$article_number](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md#property_article_number) |  | Порядковый номер товара в корзине возврата. От 1 до 999 включительно. |
| public | [$articleNumber](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md#property_articleNumber) |  | Порядковый номер товара в корзине возврата. От 1 до 999 включительно. |
| public | [$payment_article_number](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md#property_payment_article_number) |  | Порядковый номер товара в одобренной корзине покупки (article_number в [объекте платежа](https://yookassa.ru/developers/api#payment_object)). От 1 до 999 включительно. |
| public | [$paymentArticleNumber](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md#property_paymentArticleNumber) |  | Порядковый номер товара в одобренной корзине покупки (article_number в [объекте платежа](https://yookassa.ru/developers/api#payment_object)). От 1 до 999 включительно. |
| public | [$quantity](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md#property_quantity) |  | Количество возвращаемых единиц товара. Формат: целое положительное число. |
| public | [$tru_code](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md#property_tru_code) |  | Код ТРУ. 30 символов, две группы цифр, разделенные точкой. Формат: ~`NNNNNNNNN.NNNNNNNNNYYYYMMMMZZZ`, где ~`NNNNNNNNN.NNNNNNNNN` — код вида ТРУ по [Перечню ТРУ](https://esnsi.gosuslugi.ru/classifiers/10616/data?pg=1&p=1), ~`YYYY` — код производителя, ~`MMMM` — код модели, ~`ZZZ` — код страны производителя. Пример: ~`329921120.06001010200080001643`  [Как сформировать код ТРУ](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/basics#payments-preparations-tru-code) |
| public | [$truCode](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md#property_truCode) |  | Код ТРУ. 30 символов, две группы цифр, разделенные точкой. Формат: ~`NNNNNNNNN.NNNNNNNNNYYYYMMMMZZZ`, где ~`NNNNNNNNN.NNNNNNNNN` — код вида ТРУ по [Перечню ТРУ](https://esnsi.gosuslugi.ru/classifiers/10616/data?pg=1&p=1), ~`YYYY` — код производителя, ~`MMMM` — код модели, ~`ZZZ` — код страны производителя. Пример: ~`329921120.06001010200080001643`  [Как сформировать код ТРУ](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/basics#payments-preparations-tru-code) |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getArticleNumber()](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md#method_getArticleNumber) |  | Возвращает article_number. |
| public | [getPaymentArticleNumber()](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md#method_getPaymentArticleNumber) |  | Возвращает payment_article_number. |
| public | [getQuantity()](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md#method_getQuantity) |  | Возвращает quantity. |
| public | [getTruCode()](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md#method_getTruCode) |  | Возвращает tru_code. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setArticleNumber()](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md#method_setArticleNumber) |  | Устанавливает article_number. |
| public | [setPaymentArticleNumber()](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md#method_setPaymentArticleNumber) |  | Устанавливает payment_article_number. |
| public | [setQuantity()](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md#method_setQuantity) |  | Устанавливает quantity. |
| public | [setTruCode()](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md#method_setTruCode) |  | Устанавливает tru_code. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Model/Refund/RefundMethod/ElectronicCertificate/ElectronicCertificateRefundArticle.php](../../lib/Model/Refund/RefundMethod/ElectronicCertificate/ElectronicCertificateRefundArticle.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Model\Refund\RefundMethod\ElectronicCertificate\ElectronicCertificateRefundArticle

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
<a name="property_article_number"></a>
#### public $article_number : int
---
***Description***

Порядковый номер товара в корзине возврата. От 1 до 999 включительно.

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**


<a name="property_articleNumber"></a>
#### public $articleNumber : int
---
***Description***

Порядковый номер товара в корзине возврата. От 1 до 999 включительно.

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**


<a name="property_payment_article_number"></a>
#### public $payment_article_number : int
---
***Description***

Порядковый номер товара в одобренной корзине покупки (article_number в [объекте платежа](https://yookassa.ru/developers/api#payment_object)). От 1 до 999 включительно.

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**


<a name="property_paymentArticleNumber"></a>
#### public $paymentArticleNumber : int
---
***Description***

Порядковый номер товара в одобренной корзине покупки (article_number в [объекте платежа](https://yookassa.ru/developers/api#payment_object)). От 1 до 999 включительно.

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**


<a name="property_quantity"></a>
#### public $quantity : int
---
***Description***

Количество возвращаемых единиц товара. Формат: целое положительное число.

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**


<a name="property_tru_code"></a>
#### public $tru_code : string
---
***Description***

Код ТРУ. 30 символов, две группы цифр, разделенные точкой. Формат: ~`NNNNNNNNN.NNNNNNNNNYYYYMMMMZZZ`, где ~`NNNNNNNNN.NNNNNNNNN` — код вида ТРУ по [Перечню ТРУ](https://esnsi.gosuslugi.ru/classifiers/10616/data?pg=1&p=1), ~`YYYY` — код производителя, ~`MMMM` — код модели, ~`ZZZ` — код страны производителя. Пример: ~`329921120.06001010200080001643`  [Как сформировать код ТРУ](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/basics#payments-preparations-tru-code)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_truCode"></a>
#### public $truCode : string
---
***Description***

Код ТРУ. 30 символов, две группы цифр, разделенные точкой. Формат: ~`NNNNNNNNN.NNNNNNNNNYYYYMMMMZZZ`, где ~`NNNNNNNNN.NNNNNNNNN` — код вида ТРУ по [Перечню ТРУ](https://esnsi.gosuslugi.ru/classifiers/10616/data?pg=1&p=1), ~`YYYY` — код производителя, ~`MMMM` — код модели, ~`ZZZ` — код страны производителя. Пример: ~`329921120.06001010200080001643`  [Как сформировать код ТРУ](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/basics#payments-preparations-tru-code)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(array|null $data = []) : mixed
```

**Summary**

AbstractObject constructor.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR null</code> | data  |  |

**Returns:** mixed - 


<a name="method___get" class="anchor"></a>
#### public __get() : mixed

```php
public __get(string $propertyName) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method___isset" class="anchor"></a>
#### public __isset() : bool

```php
public __isset(string $propertyName) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method___set" class="anchor"></a>
#### public __set() : void

```php
public __set(string $propertyName, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method___unset" class="anchor"></a>
#### public __unset() : void

```php
public __unset(string $propertyName) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_fromArray" class="anchor"></a>
#### public fromArray() : void

```php
public fromArray(array|\Traversable $sourceArray) : void
```

**Summary**

Устанавливает значения свойств текущего объекта из массива.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \Traversable</code> | sourceArray  | Ассоциативный массив с настройками |

**Returns:** void - 


<a name="method_getArticleNumber" class="anchor"></a>
#### public getArticleNumber() : int|null

```php
public getArticleNumber() : int|null
```

**Summary**

Возвращает article_number.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\RefundMethod\ElectronicCertificate\ElectronicCertificateRefundArticle](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md)

**Returns:** int|null - Порядковый номер товара в корзине возврата


<a name="method_getPaymentArticleNumber" class="anchor"></a>
#### public getPaymentArticleNumber() : int|null

```php
public getPaymentArticleNumber() : int|null
```

**Summary**

Возвращает payment_article_number.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\RefundMethod\ElectronicCertificate\ElectronicCertificateRefundArticle](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md)

**Returns:** int|null - Порядковый номер товара в одобренной корзине покупки


<a name="method_getQuantity" class="anchor"></a>
#### public getQuantity() : int|null

```php
public getQuantity() : int|null
```

**Summary**

Возвращает quantity.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\RefundMethod\ElectronicCertificate\ElectronicCertificateRefundArticle](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md)

**Returns:** int|null - Количество возвращаемых единиц товара


<a name="method_getTruCode" class="anchor"></a>
#### public getTruCode() : string|null

```php
public getTruCode() : string|null
```

**Summary**

Возвращает tru_code.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\RefundMethod\ElectronicCertificate\ElectronicCertificateRefundArticle](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md)

**Returns:** string|null - Код ТРУ


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_jsonSerialize" class="anchor"></a>
#### public jsonSerialize() : array

```php
public jsonSerialize() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_offsetExists" class="anchor"></a>
#### public offsetExists() : bool

```php
public offsetExists(string $offset) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method_offsetGet" class="anchor"></a>
#### public offsetGet() : mixed

```php
public offsetGet(string $offset) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method_offsetSet" class="anchor"></a>
#### public offsetSet() : void

```php
public offsetSet(string $offset, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method_offsetUnset" class="anchor"></a>
#### public offsetUnset() : void

```php
public offsetUnset(string $offset) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_setArticleNumber" class="anchor"></a>
#### public setArticleNumber() : self

```php
public setArticleNumber(int|null $article_number = null) : self
```

**Summary**

Устанавливает article_number.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\RefundMethod\ElectronicCertificate\ElectronicCertificateRefundArticle](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int OR null</code> | article_number  | Порядковый номер товара в корзине возврата |

**Returns:** self - 


<a name="method_setPaymentArticleNumber" class="anchor"></a>
#### public setPaymentArticleNumber() : self

```php
public setPaymentArticleNumber(int|null $payment_article_number = null) : self
```

**Summary**

Устанавливает payment_article_number.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\RefundMethod\ElectronicCertificate\ElectronicCertificateRefundArticle](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int OR null</code> | payment_article_number  | Порядковый номер товара в одобренной корзине покупки |

**Returns:** self - 


<a name="method_setQuantity" class="anchor"></a>
#### public setQuantity() : self

```php
public setQuantity(int|null $quantity = null) : self
```

**Summary**

Устанавливает quantity.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\RefundMethod\ElectronicCertificate\ElectronicCertificateRefundArticle](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int OR null</code> | quantity  | Количество возвращаемых единиц товара. Формат: целое положительное число. |

**Returns:** self - 


<a name="method_setTruCode" class="anchor"></a>
#### public setTruCode() : self

```php
public setTruCode(string|null $tru_code = null) : self
```

**Summary**

Устанавливает tru_code.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\RefundMethod\ElectronicCertificate\ElectronicCertificateRefundArticle](../classes/YooKassa-Model-Refund-RefundMethod-ElectronicCertificate-ElectronicCertificateRefundArticle.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | tru_code  | Код ТРУ. 30 символов, две группы цифр, разделенные точкой. Формат: ~`NNNNNNNNN.NNNNNNNNNYYYYMMMMZZZ`, где ~`NNNNNNNNN.NNNNNNNNN` — код вида ТРУ по [Перечню ТРУ](https://esnsi.gosuslugi.ru/classifiers/10616/data?pg=1&p=1), ~`YYYY` — код производителя, ~`MMMM` — код модели, ~`ZZZ` — код страны производителя. Пример: ~`329921120.06001010200080001643`  [Как сформировать код ТРУ](/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate#payments-preparation-tru-code) |

**Returns:** self - 


<a name="method_toArray" class="anchor"></a>
#### public toArray() : array

```php
public toArray() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации
Является алиасом метода AbstractObject::jsonSerialize().

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_getUnknownProperties" class="anchor"></a>
#### protected getUnknownProperties() : array

```php
protected getUnknownProperties() : array
```

**Summary**

Возвращает массив свойств которые не существуют, но были заданы у объекта.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив с не существующими у текущего объекта свойствами


<a name="method_validatePropertyValue" class="anchor"></a>
#### protected validatePropertyValue() : mixed

```php
protected validatePropertyValue(string $propertyName, mixed $propertyValue) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  |  |
| <code lang="php">mixed</code> | propertyValue  |  |

**Returns:** mixed - 



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