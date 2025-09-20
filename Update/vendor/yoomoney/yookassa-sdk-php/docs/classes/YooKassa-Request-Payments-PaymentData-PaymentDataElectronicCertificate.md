# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payments\PaymentData\PaymentDataElectronicCertificate
### Namespace: [\YooKassa\Request\Payments\PaymentData](../namespaces/yookassa-request-payments-paymentdata.md)
---
**Summary:**

Класс, представляющий модель PaymentMethodDataElectronicCertificate.

**Description:**

Данные для оплаты по электронному сертификату.

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$articles](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataElectronicCertificate.md#property_articles) |  | Корзина покупки (в терминах НСПК) — список товаров, которые можно оплатить по сертификату.  Необходимо передавать только при [оплате на готовой странице ЮKassa](/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/ready-made-payment-form). |
| public | [$card](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataElectronicCertificate.md#property_card) |  | Данные банковской карты «Мир». |
| public | [$electronic_certificate](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataElectronicCertificate.md#property_electronic_certificate) |  | Данные от ФЭС НСПК для оплаты по электронному сертификату. |
| public | [$electronicCertificate](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataElectronicCertificate.md#property_electronicCertificate) |  | Данные от ФЭС НСПК для оплаты по электронному сертификату. |
| public | [$type](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataElectronicCertificate.md#property_type) |  | Код способа оплаты. |
| public | [$type](../classes/YooKassa-Request-Payments-PaymentData-AbstractPaymentData.md#property_type) |  | Тип метода оплаты |
| protected | [$_type](../classes/YooKassa-Request-Payments-PaymentData-AbstractPaymentData.md#property__type) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataElectronicCertificate.md#method___construct) |  |  |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getArticles()](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataElectronicCertificate.md#method_getArticles) |  | Возвращает articles. |
| public | [getCard()](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataElectronicCertificate.md#method_getCard) |  | Возвращает card. |
| public | [getElectronicCertificate()](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataElectronicCertificate.md#method_getElectronicCertificate) |  | Возвращает electronic_certificate. |
| public | [getType()](../classes/YooKassa-Request-Payments-PaymentData-AbstractPaymentData.md#method_getType) |  | Возвращает тип метода оплаты. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setArticles()](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataElectronicCertificate.md#method_setArticles) |  | Устанавливает articles. |
| public | [setCard()](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataElectronicCertificate.md#method_setCard) |  | Устанавливает card. |
| public | [setElectronicCertificate()](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataElectronicCertificate.md#method_setElectronicCertificate) |  | Устанавливает electronic_certificate. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [setType()](../classes/YooKassa-Request-Payments-PaymentData-AbstractPaymentData.md#method_setType) |  | Устанавливает тип метода оплаты. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Payments/PaymentData/PaymentDataElectronicCertificate.php](../../lib/Request/Payments/PaymentData/PaymentDataElectronicCertificate.php)
* Package: YooKassa\Model
* Class Hierarchy:  
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Request\Payments\PaymentData\AbstractPaymentData](../classes/YooKassa-Request-Payments-PaymentData-AbstractPaymentData.md)
  * \YooKassa\Request\Payments\PaymentData\PaymentDataElectronicCertificate

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
<a name="property_articles"></a>
#### public $articles : \YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle[]|\YooKassa\Common\ListObjectInterface|null
---
***Description***

Корзина покупки (в терминах НСПК) — список товаров, которые можно оплатить по сертификату.  Необходимо передавать только при [оплате на готовой странице ЮKassa](/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/ready-made-payment-form).

**Type:** <a href="../\YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle[]|\YooKassa\Common\ListObjectInterface|null"><abbr title="\YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle[]|\YooKassa\Common\ListObjectInterface|null">ListObjectInterface|null</abbr></a>

**Details:**


<a name="property_card"></a>
#### public $card : \YooKassa\Request\Payments\PaymentData\PaymentDataBankCardCard|null
---
***Description***

Данные банковской карты «Мир».

**Type:** <a href="../\YooKassa\Request\Payments\PaymentData\PaymentDataBankCardCard|null"><abbr title="\YooKassa\Request\Payments\PaymentData\PaymentDataBankCardCard|null">PaymentDataBankCardCard|null</abbr></a>

**Details:**


<a name="property_electronic_certificate"></a>
#### public $electronic_certificate : \YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificatePaymentData|null
---
***Description***

Данные от ФЭС НСПК для оплаты по электронному сертификату.

**Type:** <a href="../\YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificatePaymentData|null"><abbr title="\YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificatePaymentData|null">ElectronicCertificatePaymentData|null</abbr></a>

**Details:**


<a name="property_electronicCertificate"></a>
#### public $electronicCertificate : \YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificatePaymentData|null
---
***Description***

Данные от ФЭС НСПК для оплаты по электронному сертификату.

**Type:** <a href="../\YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificatePaymentData|null"><abbr title="\YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificatePaymentData|null">ElectronicCertificatePaymentData|null</abbr></a>

**Details:**


<a name="property_type"></a>
#### public $type : string
---
***Description***

Код способа оплаты.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_type"></a>
#### public $type : string
---
***Description***

Тип метода оплаты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\AbstractPaymentData](../classes/YooKassa-Request-Payments-PaymentData-AbstractPaymentData.md)


<a name="property__type"></a>
#### protected $_type : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\AbstractPaymentData](../classes/YooKassa-Request-Payments-PaymentData-AbstractPaymentData.md)



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(?array $data = []) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\PaymentDataElectronicCertificate](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataElectronicCertificate.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">?array</code> | data  |  |

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


<a name="method_getArticles" class="anchor"></a>
#### public getArticles() : \YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle[]|\YooKassa\Common\ListObjectInterface|null

```php
public getArticles() : \YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle[]|\YooKassa\Common\ListObjectInterface|null
```

**Summary**

Возвращает articles.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\PaymentDataElectronicCertificate](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataElectronicCertificate.md)

**Returns:** \YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle[]|\YooKassa\Common\ListObjectInterface|null - Корзина покупки (в терминах НСПК) — список товаров


<a name="method_getCard" class="anchor"></a>
#### public getCard() : \YooKassa\Request\Payments\PaymentData\PaymentDataBankCardCard|null

```php
public getCard() : \YooKassa\Request\Payments\PaymentData\PaymentDataBankCardCard|null
```

**Summary**

Возвращает card.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\PaymentDataElectronicCertificate](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataElectronicCertificate.md)

**Returns:** \YooKassa\Request\Payments\PaymentData\PaymentDataBankCardCard|null - Данные банковской карты «Мир».


<a name="method_getElectronicCertificate" class="anchor"></a>
#### public getElectronicCertificate() : \YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificatePaymentData|null

```php
public getElectronicCertificate() : \YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificatePaymentData|null
```

**Summary**

Возвращает electronic_certificate.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\PaymentDataElectronicCertificate](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataElectronicCertificate.md)

**Returns:** \YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificatePaymentData|null - Данные от ФЭС НСПК для оплаты по электронному сертификату.


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает тип метода оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\AbstractPaymentData](../classes/YooKassa-Request-Payments-PaymentData-AbstractPaymentData.md)

**Returns:** string|null - Тип метода оплаты


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


<a name="method_setArticles" class="anchor"></a>
#### public setArticles() : self

```php
public setArticles(\YooKassa\Common\ListObjectInterface|array|null $articles = null) : self
```

**Summary**

Устанавливает articles.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\PaymentDataElectronicCertificate](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataElectronicCertificate.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Common\ListObjectInterface OR array OR null</code> | articles  | Корзина покупки (в терминах НСПК) — список товаров, которые можно оплатить по сертификату.  Необходимо передавать только при [оплате на готовой странице ЮKassa](/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/ready-made-payment-form). |

**Returns:** self - 


<a name="method_setCard" class="anchor"></a>
#### public setCard() : self

```php
public setCard(\YooKassa\Request\Payments\PaymentData\PaymentDataBankCardCard|array|null $card = null) : self
```

**Summary**

Устанавливает card.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\PaymentDataElectronicCertificate](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataElectronicCertificate.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Payments\PaymentData\PaymentDataBankCardCard OR array OR null</code> | card  | Данные банковской карты «Мир». |

**Returns:** self - 


<a name="method_setElectronicCertificate" class="anchor"></a>
#### public setElectronicCertificate() : self

```php
public setElectronicCertificate(\YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificatePaymentData|array|null $electronic_certificate = null) : self
```

**Summary**

Устанавливает electronic_certificate.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\PaymentDataElectronicCertificate](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataElectronicCertificate.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificatePaymentData OR array OR null</code> | electronic_certificate  | Данные от ФЭС НСПК для оплаты по электронному сертификату. |

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


<a name="method_setType" class="anchor"></a>
#### protected setType() : self

```php
protected setType(string|null $type) : self
```

**Summary**

Устанавливает тип метода оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\AbstractPaymentData](../classes/YooKassa-Request-Payments-PaymentData-AbstractPaymentData.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Тип метода оплаты |

**Returns:** self - 


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