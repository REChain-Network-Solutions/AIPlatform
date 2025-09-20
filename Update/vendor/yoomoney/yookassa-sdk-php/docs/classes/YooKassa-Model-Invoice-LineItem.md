# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Invoice\LineItem
### Namespace: [\YooKassa\Model\Invoice](../namespaces/yookassa-model-invoice.md)
---
**Summary:**

Класс, представляющий модель LineItem.

**Description:**

Данные о товаре или услуге в корзине.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [MIN_LENGTH_DESCRIPTION](../classes/YooKassa-Model-Invoice-LineItem.md#constant_MIN_LENGTH_DESCRIPTION) |  |  |
| public | [MAX_LENGTH_DESCRIPTION](../classes/YooKassa-Model-Invoice-LineItem.md#constant_MAX_LENGTH_DESCRIPTION) |  |  |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$description](../classes/YooKassa-Model-Invoice-LineItem.md#property_description) |  | Название товара или услуги (от 1 до 128 символов). Пользователь увидит его на странице счета перед оплатой. |
| public | [$discount_price](../classes/YooKassa-Model-Invoice-LineItem.md#property_discount_price) |  | Итоговая цена товара с учетом скидки. Если передана, то на странице счета цена отобразится с учетом скидки. Не нужно передавать, если пользователь оплачивает полную стоимость товара или услуги. |
| public | [$discountPrice](../classes/YooKassa-Model-Invoice-LineItem.md#property_discountPrice) |  | Итоговая цена товара с учетом скидки. Если передана, то на странице счета цена отобразится с учетом скидки. Не нужно передавать, если пользователь оплачивает полную стоимость товара или услуги. |
| public | [$price](../classes/YooKassa-Model-Invoice-LineItem.md#property_price) |  | Полная цена товара или услуги. Пользователь увидит ее на странице счета перед оплатой. |
| public | [$quantity](../classes/YooKassa-Model-Invoice-LineItem.md#property_quantity) |  | Количество товара. Можно передать целое или дробное число. Разделитель дробной части — точка, разделитель тысяч отсутствует, максимум три знака после точки. Пример: ~`5.000` |

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
| public | [getDescription()](../classes/YooKassa-Model-Invoice-LineItem.md#method_getDescription) |  | Возвращает description. |
| public | [getDiscountPrice()](../classes/YooKassa-Model-Invoice-LineItem.md#method_getDiscountPrice) |  | Возвращает discount_price. |
| public | [getPrice()](../classes/YooKassa-Model-Invoice-LineItem.md#method_getPrice) |  | Возвращает price. |
| public | [getQuantity()](../classes/YooKassa-Model-Invoice-LineItem.md#method_getQuantity) |  | Возвращает quantity. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setDescription()](../classes/YooKassa-Model-Invoice-LineItem.md#method_setDescription) |  | Устанавливает description. |
| public | [setDiscountPrice()](../classes/YooKassa-Model-Invoice-LineItem.md#method_setDiscountPrice) |  | Устанавливает discount_price. |
| public | [setPrice()](../classes/YooKassa-Model-Invoice-LineItem.md#method_setPrice) |  | Устанавливает price. |
| public | [setQuantity()](../classes/YooKassa-Model-Invoice-LineItem.md#method_setQuantity) |  | Устанавливает quantity. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Model/Invoice/LineItem.php](../../lib/Model/Invoice/LineItem.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Model\Invoice\LineItem

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
<a name="constant_MIN_LENGTH_DESCRIPTION" class="anchor"></a>
###### MIN_LENGTH_DESCRIPTION
```php
MIN_LENGTH_DESCRIPTION = 1 : int
```


<a name="constant_MAX_LENGTH_DESCRIPTION" class="anchor"></a>
###### MAX_LENGTH_DESCRIPTION
```php
MAX_LENGTH_DESCRIPTION = 128 : int
```



---
## Properties
<a name="property_description"></a>
#### public $description : string
---
***Description***

Название товара или услуги (от 1 до 128 символов). Пользователь увидит его на странице счета перед оплатой.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_discount_price"></a>
#### public $discount_price : \YooKassa\Model\AmountInterface|null
---
***Description***

Итоговая цена товара с учетом скидки. Если передана, то на странице счета цена отобразится с учетом скидки. Не нужно передавать, если пользователь оплачивает полную стоимость товара или услуги.

**Type:** <a href="../\YooKassa\Model\AmountInterface|null"><abbr title="\YooKassa\Model\AmountInterface|null">AmountInterface|null</abbr></a>

**Details:**


<a name="property_discountPrice"></a>
#### public $discountPrice : \YooKassa\Model\AmountInterface|null
---
***Description***

Итоговая цена товара с учетом скидки. Если передана, то на странице счета цена отобразится с учетом скидки. Не нужно передавать, если пользователь оплачивает полную стоимость товара или услуги.

**Type:** <a href="../\YooKassa\Model\AmountInterface|null"><abbr title="\YooKassa\Model\AmountInterface|null">AmountInterface|null</abbr></a>

**Details:**


<a name="property_price"></a>
#### public $price : \YooKassa\Model\AmountInterface
---
***Description***

Полная цена товара или услуги. Пользователь увидит ее на странице счета перед оплатой.

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_quantity"></a>
#### public $quantity : float
---
***Description***

Количество товара. Можно передать целое или дробное число. Разделитель дробной части — точка, разделитель тысяч отсутствует, максимум три знака после точки. Пример: ~`5.000`

**Type:** <a href="../float"><abbr title="float">float</abbr></a>

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


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает description.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\LineItem](../classes/YooKassa-Model-Invoice-LineItem.md)

**Returns:** string|null - Название товара или услуги


<a name="method_getDiscountPrice" class="anchor"></a>
#### public getDiscountPrice() : \YooKassa\Model\AmountInterface|null

```php
public getDiscountPrice() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает discount_price.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\LineItem](../classes/YooKassa-Model-Invoice-LineItem.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Итоговая цена товара с учетом скидки


<a name="method_getPrice" class="anchor"></a>
#### public getPrice() : \YooKassa\Model\AmountInterface|null

```php
public getPrice() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает price.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\LineItem](../classes/YooKassa-Model-Invoice-LineItem.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Полная цена товара или услуги


<a name="method_getQuantity" class="anchor"></a>
#### public getQuantity() : float|null

```php
public getQuantity() : float|null
```

**Summary**

Возвращает quantity.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\LineItem](../classes/YooKassa-Model-Invoice-LineItem.md)

**Returns:** float|null - Количество товара


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


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : self

```php
public setDescription(string|null $description = null) : self
```

**Summary**

Устанавливает description.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\LineItem](../classes/YooKassa-Model-Invoice-LineItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | description  | Название товара или услуги (от 1 до 128 символов). Пользователь увидит его на странице счета перед оплатой. |

**Returns:** self - 


<a name="method_setDiscountPrice" class="anchor"></a>
#### public setDiscountPrice() : self

```php
public setDiscountPrice(\YooKassa\Model\AmountInterface|array|null $discount_price = null) : self
```

**Summary**

Устанавливает discount_price.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\LineItem](../classes/YooKassa-Model-Invoice-LineItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | discount_price  | Итоговая цена товара с учетом скидки |

**Returns:** self - 


<a name="method_setPrice" class="anchor"></a>
#### public setPrice() : self

```php
public setPrice(\YooKassa\Model\AmountInterface|array|null $price = null) : self
```

**Summary**

Устанавливает price.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\LineItem](../classes/YooKassa-Model-Invoice-LineItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | price  | Полная цена товара или услуги |

**Returns:** self - 


<a name="method_setQuantity" class="anchor"></a>
#### public setQuantity() : self

```php
public setQuantity(float|null $quantity = null) : self
```

**Summary**

Устанавливает quantity.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\LineItem](../classes/YooKassa-Model-Invoice-LineItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">float OR null</code> | quantity  | Количество товара. Можно передать целое или дробное число. Разделитель дробной части — точка, разделитель тысяч отсутствует, максимум три знака после точки. Пример: ~`5.000` |

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