# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Payment\PaymentMethod\BankCard
### Namespace: [\YooKassa\Model\Payment\PaymentMethod](../namespaces/yookassa-model-payment-paymentmethod.md)
---
**Summary:**

Класс, описывающий модель BankCard.

**Description:**

Объект банковской карты

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [ISO_3166_CODE_LENGTH](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#constant_ISO_3166_CODE_LENGTH) |  |  |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$card_product](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#property_card_product) |  | Карточный продукт платежной системы, с которым ассоциирована банковская карта. |
| public | [$card_type](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#property_card_type) |  | Тип банковской карты |
| public | [$cardProduct](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#property_cardProduct) |  | Карточный продукт платежной системы, с которым ассоциирована банковская карта. |
| public | [$cardType](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#property_cardType) |  | Тип банковской карты |
| public | [$expiry_month](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#property_expiry_month) |  | Срок действия, месяц |
| public | [$expiry_year](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#property_expiry_year) |  | Срок действия, год |
| public | [$expiryMonth](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#property_expiryMonth) |  | Срок действия, месяц |
| public | [$expiryYear](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#property_expiryYear) |  | Срок действия, год |
| public | [$first6](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#property_first6) |  | Первые 6 цифр номера карты |
| public | [$issuer_country](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#property_issuer_country) |  | Код страны, в которой выпущена карта |
| public | [$issuer_name](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#property_issuer_name) |  | Тип банковской карты |
| public | [$issuerCountry](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#property_issuerCountry) |  | Код страны, в которой выпущена карта |
| public | [$issuerName](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#property_issuerName) |  | Тип банковской карты |
| public | [$last4](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#property_last4) |  | Последние 4 цифры номера карты |
| public | [$source](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#property_source) |  | Тип банковской карты |

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
| public | [getCardProduct()](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#method_getCardProduct) |  | Возвращает карточный продукт платежной системы. |
| public | [getCardType()](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#method_getCardType) |  | Возвращает тип банковской карты. |
| public | [getExpiryMonth()](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#method_getExpiryMonth) |  | Возвращает срок действия, месяц, MM. |
| public | [getExpiryYear()](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#method_getExpiryYear) |  | Возвращает срок действия, год, YYYY. |
| public | [getFirst6()](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#method_getFirst6) |  | Возвращает первые 6 цифр номера карты (BIN). |
| public | [getIssuerCountry()](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#method_getIssuerCountry) |  | Возвращает код страны, в которой выпущена карта. Передается в формате ISO-3166 alpha-2. |
| public | [getIssuerName()](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#method_getIssuerName) |  | Возвращает наименование банка, выпустившего карту. |
| public | [getLast4()](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#method_getLast4) |  | Возвращает последние 4 цифры номера карты. |
| public | [getSource()](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#method_getSource) |  | Возвращает источник данных банковской карты. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setCardProduct()](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#method_setCardProduct) |  | Устанавливает карточный продукт платежной системы. |
| public | [setCardType()](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#method_setCardType) |  | Устанавливает тип банковской карты. |
| public | [setExpiryMonth()](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#method_setExpiryMonth) |  | Устанавливает срок действия, месяц, MM. |
| public | [setExpiryYear()](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#method_setExpiryYear) |  | Устанавливает срок действия, год, YYYY. |
| public | [setFirst6()](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#method_setFirst6) |  | Устанавливает первые 6 цифр номера карты (BIN). |
| public | [setIssuerCountry()](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#method_setIssuerCountry) |  | Устанавливает код страны, в которой выпущена карта. |
| public | [setIssuerName()](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#method_setIssuerName) |  | Устанавливает наименование банка, выпустившего карту. |
| public | [setLast4()](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#method_setLast4) |  | Устанавливает последние 4 цифры номера карты. |
| public | [setSource()](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md#method_setSource) |  | Устанавливает источник данных банковской карты. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Model/Payment/PaymentMethod/BankCard.php](../../lib/Model/Payment/PaymentMethod/BankCard.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Model\Payment\PaymentMethod\BankCard

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
<a name="constant_ISO_3166_CODE_LENGTH" class="anchor"></a>
###### ISO_3166_CODE_LENGTH
```php
ISO_3166_CODE_LENGTH = 2 : int
```



---
## Properties
<a name="property_card_product"></a>
#### public $card_product : string
---
***Description***

Карточный продукт платежной системы, с которым ассоциирована банковская карта.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_card_type"></a>
#### public $card_type : string
---
***Description***

Тип банковской карты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_cardProduct"></a>
#### public $cardProduct : string
---
***Description***

Карточный продукт платежной системы, с которым ассоциирована банковская карта.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_cardType"></a>
#### public $cardType : string
---
***Description***

Тип банковской карты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_expiry_month"></a>
#### public $expiry_month : string
---
***Description***

Срок действия, месяц

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_expiry_year"></a>
#### public $expiry_year : string
---
***Description***

Срок действия, год

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_expiryMonth"></a>
#### public $expiryMonth : string
---
***Description***

Срок действия, месяц

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_expiryYear"></a>
#### public $expiryYear : string
---
***Description***

Срок действия, год

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_first6"></a>
#### public $first6 : string
---
***Description***

Первые 6 цифр номера карты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_issuer_country"></a>
#### public $issuer_country : string
---
***Description***

Код страны, в которой выпущена карта

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_issuer_name"></a>
#### public $issuer_name : string
---
***Description***

Тип банковской карты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_issuerCountry"></a>
#### public $issuerCountry : string
---
***Description***

Код страны, в которой выпущена карта

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_issuerName"></a>
#### public $issuerName : string
---
***Description***

Тип банковской карты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_last4"></a>
#### public $last4 : string
---
***Description***

Последние 4 цифры номера карты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_source"></a>
#### public $source : string
---
***Description***

Тип банковской карты

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


<a name="method_getCardProduct" class="anchor"></a>
#### public getCardProduct() : \YooKassa\Model\Payment\PaymentMethod\BankCardProduct|null

```php
public getCardProduct() : \YooKassa\Model\Payment\PaymentMethod\BankCardProduct|null
```

**Summary**

Возвращает карточный продукт платежной системы.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\BankCard](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md)

**Returns:** \YooKassa\Model\Payment\PaymentMethod\BankCardProduct|null - Карточный продукт платежной системы


<a name="method_getCardType" class="anchor"></a>
#### public getCardType() : string|null

```php
public getCardType() : string|null
```

**Summary**

Возвращает тип банковской карты.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\BankCard](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md)

**Returns:** string|null - Тип банковской карты


<a name="method_getExpiryMonth" class="anchor"></a>
#### public getExpiryMonth() : string|null

```php
public getExpiryMonth() : string|null
```

**Summary**

Возвращает срок действия, месяц, MM.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\BankCard](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md)

**Returns:** string|null - Срок действия, месяц, MM


<a name="method_getExpiryYear" class="anchor"></a>
#### public getExpiryYear() : string|null

```php
public getExpiryYear() : string|null
```

**Summary**

Возвращает срок действия, год, YYYY.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\BankCard](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md)

**Returns:** string|null - Срок действия, год, YYYY


<a name="method_getFirst6" class="anchor"></a>
#### public getFirst6() : string|null

```php
public getFirst6() : string|null
```

**Summary**

Возвращает первые 6 цифр номера карты (BIN).

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\BankCard](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md)

**Returns:** string|null - Первые 6 цифр номера карты (BIN)


<a name="method_getIssuerCountry" class="anchor"></a>
#### public getIssuerCountry() : string|null

```php
public getIssuerCountry() : string|null
```

**Summary**

Возвращает код страны, в которой выпущена карта. Передается в формате ISO-3166 alpha-2.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\BankCard](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md)

**Returns:** string|null - Код страны, в которой выпущена карта


<a name="method_getIssuerName" class="anchor"></a>
#### public getIssuerName() : string|null

```php
public getIssuerName() : string|null
```

**Summary**

Возвращает наименование банка, выпустившего карту.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\BankCard](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md)

**Returns:** string|null - Наименование банка, выпустившего карту


<a name="method_getLast4" class="anchor"></a>
#### public getLast4() : string|null

```php
public getLast4() : string|null
```

**Summary**

Возвращает последние 4 цифры номера карты.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\BankCard](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md)

**Returns:** string|null - Последние 4 цифры номера карты


<a name="method_getSource" class="anchor"></a>
#### public getSource() : string|null

```php
public getSource() : string|null
```

**Summary**

Возвращает источник данных банковской карты.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\BankCard](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md)

**Returns:** string|null - Источник данных банковской карты


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


<a name="method_setCardProduct" class="anchor"></a>
#### public setCardProduct() : self

```php
public setCardProduct(\YooKassa\Model\Payment\PaymentMethod\BankCardProduct|array|null $card_product = null) : self
```

**Summary**

Устанавливает карточный продукт платежной системы.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\BankCard](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Payment\PaymentMethod\BankCardProduct OR array OR null</code> | card_product  | Карточный продукт платежной системы |

**Returns:** self - 


<a name="method_setCardType" class="anchor"></a>
#### public setCardType() : self

```php
public setCardType(string|null $card_type = null) : self
```

**Summary**

Устанавливает тип банковской карты.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\BankCard](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | card_type  | Тип банковской карты |

**Returns:** self - 


<a name="method_setExpiryMonth" class="anchor"></a>
#### public setExpiryMonth() : self

```php
public setExpiryMonth(string|null $expiry_month = null) : self
```

**Summary**

Устанавливает срок действия, месяц, MM.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\BankCard](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | expiry_month  | Срок действия, месяц, MM. |

**Returns:** self - 


<a name="method_setExpiryYear" class="anchor"></a>
#### public setExpiryYear() : self

```php
public setExpiryYear(string|null $expiry_year = null) : self
```

**Summary**

Устанавливает срок действия, год, YYYY.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\BankCard](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | expiry_year  | Срок действия, год, YYYY. |

**Returns:** self - 


<a name="method_setFirst6" class="anchor"></a>
#### public setFirst6() : self

```php
public setFirst6(string|null $first6 = null) : self
```

**Summary**

Устанавливает первые 6 цифр номера карты (BIN).

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\BankCard](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | first6  | Первые 6 цифр номера карты (BIN). При оплате картой, [сохраненной в ЮKassa](/developers/payment-acceptance/scenario-extensions/recurring-payments) и других сервисах, переданный BIN может не соответствовать значениям `last4`, `expiry_year`, `expiry_month`. |

**Returns:** self - 


<a name="method_setIssuerCountry" class="anchor"></a>
#### public setIssuerCountry() : self

```php
public setIssuerCountry(string|null $issuer_country = null) : self
```

**Summary**

Устанавливает код страны, в которой выпущена карта.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\BankCard](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | issuer_country  | Код страны, в которой выпущена карта. Передается в формате [ISO-3166 alpha-2](https://www.iso.org/obp/ui/#iso:pub:PUB500001:en). Пример: ~`RU`. |

**Returns:** self - 


<a name="method_setIssuerName" class="anchor"></a>
#### public setIssuerName() : self

```php
public setIssuerName(string|null $issuer_name = null) : self
```

**Summary**

Устанавливает наименование банка, выпустившего карту.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\BankCard](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | issuer_name  | Наименование банка, выпустившего карту. |

**Returns:** self - 


<a name="method_setLast4" class="anchor"></a>
#### public setLast4() : self

```php
public setLast4(string|null $last4 = null) : self
```

**Summary**

Устанавливает последние 4 цифры номера карты.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\BankCard](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | last4  | Последние 4 цифры номера карты. |

**Returns:** self - 


<a name="method_setSource" class="anchor"></a>
#### public setSource() : self

```php
public setSource(string|null $source = null) : self
```

**Summary**

Устанавливает источник данных банковской карты.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\BankCard](../classes/YooKassa-Model-Payment-PaymentMethod-BankCard.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | source  | Источник данных банковской карты |

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