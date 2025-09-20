<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace YooKassa\Request\Deals;

use YooKassa\Common\AbstractRequestBuilder;
use YooKassa\Common\AbstractRequestInterface;
use YooKassa\Model\Metadata;

/**
 * Класс, представляющий модель CreateDealRequestBuilder.
 *
 * Класс билдера объектов запросов к API на создание платежа.
 *
 * @example 02-builder.php 252 19 Пример использования билдера
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class CreateDealRequestBuilder extends AbstractRequestBuilder
{
    /**
     * Собираемый объект запроса.
     *
     * @var CreateDealRequest|null
     */
    protected ?AbstractRequestInterface $currentObject = null;

    /**
     * Устанавливает тип сделки.
     *
     * @param string $value Тип сделки
     *
     * @return CreateDealRequestBuilder Инстанс текущего билдера
     */
    public function setType(string $value): CreateDealRequestBuilder
    {
        $this->currentObject->setType($value);

        return $this;
    }

    /**
     * Устанавливает момент перечисления вам вознаграждения платформы.
     *
     * @param string $value Момент перечисления вам вознаграждения платформы
     *
     * @return CreateDealRequestBuilder Инстанс текущего билдера
     */
    public function setFeeMoment(string $value): CreateDealRequestBuilder
    {
        $this->currentObject->setFeeMoment($value);

        return $this;
    }

    /**
     * Устанавливает метаданные, привязанные к платежу.
     *
     * @param null|array|Metadata $value Метаданные платежа, устанавливаемые мерчантом
     *
     * @return CreateDealRequestBuilder Инстанс текущего билдера
     */
    public function setMetadata(mixed $value): CreateDealRequestBuilder
    {
        $this->currentObject->setMetadata($value);

        return $this;
    }

    /**
     * Устанавливает описание транзакции.
     *
     * @param string|null $value Описание транзакции
     *
     * @return CreateDealRequestBuilder Инстанс текущего билдера
     */
    public function setDescription(?string $value): CreateDealRequestBuilder
    {
        $this->currentObject->setDescription($value);

        return $this;
    }

    /**
     * Осуществляет сборку объекта запроса к API.
     *
     * @param array|null $options
     *
     * @return CreateDealRequestInterface|AbstractRequestInterface
     */
    public function build(?array $options = null): AbstractRequestInterface
    {
        return parent::build($options);
    }

    /**
     * Инициализирует объект запроса, который в дальнейшем будет собираться билдером
     *
     * @return CreateDealRequest Инстанс собираемого объекта запроса к API
     */
    protected function initCurrentObject(): CreateDealRequest
    {
        return new CreateDealRequest();
    }
}
